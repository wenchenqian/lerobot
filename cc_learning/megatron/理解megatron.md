# 整体框架

![img_1.png](img_1.png)
整体框架如上图所示，训练和推理各自有其文件夹，文件夹下有对应的训练和推理api，然后core文件夹内有这最核心的一些组件的实现，如optimizer、models、Transformer和各种并行和分布式通讯的库。

# 并行初始化

megatron的并行初始化调用的是initialize_megatron()，内部会调用initialize_model_parallel()，接下来我们看看这个函数是如何对DP、PP、TP、CP、EP进行通讯域初始化的：

```
""" 函数的参数太多了，这里省略 """
def initialize_model_parallel(...) -> None:
    """ 初始化DP并行组 """
    ... # 全局变量获取
    for ranks in generator_wrapper('dp'):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs))
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
        if rank in ranks:
            """ 
            这里的if判断的逻辑是：只有我这个训练进程的rank在这个group里我才会把我这个训练进程的并行通信组设置为刚才创建出来的group。
            而这个group的类型是pytorch的ProcessGroup对象，对于下面所有的并行组的创建都是走的这个逻辑。
            """
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GROUP_GLOO = group_gloo
            _DATA_PARALLEL_GLOBAL_RANKS = ranks
  
    """ 初始化DP-CP并行组，由于CP的并行逻辑原因，DP和CP的分组可以共用一个通信组 """
    for ranks_with_cp in generator_wrapper('dp-cp'):
        group_with_cp = torch.distributed.new_group(ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs))
        group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, timeout=timeout, backend="gloo")
        if rank in ranks_with_cp:
             # do samething
   
    ...

    """ 初始化CP并行组 """
    ... # 全局变量获取
    for ranks in generator_wrapper('cp'):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs))
        if rank in ranks:
            # do samething

    """ 初始化DP域并行组，这个group内包含了一个完整模型参数所需的rank号 """
    ... # 全局变量获取
    for ranks in generator_wrapper('tp-pp'):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs))
        if rank in ranks:
            # do samething
    ...
   
    """ 初始化TP并行组 """
    ... # 全局变量获取
    for ranks in generator_wrapper('tp'):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs))
        if rank in ranks:
            # do samething

    """ 初始化PP并行组 """
    ... # 全局变量获取
    for ranks in generator_wrapper('pp'):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs))
        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP is None:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                _PIPELINE_GLOBAL_RANKS.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP = [_PIPELINE_MODEL_PARALLEL_GROUP, group]
                _PIPELINE_GLOBAL_RANKS = [_PIPELINE_GLOBAL_RANKS, ranks]

        embedding_ranks = get_embedding_ranks(ranks)
        group = torch.distributed.new_group(embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs))
        if rank in embedding_ranks:
            # do samething

        position_embedding_ranks = get_position_embedding_ranks(ranks)
        group = torch.distributed.new_group(...)
        if rank in position_embedding_ranks:
            # do samething
   
    """ 初始化TP+CP并行组，这个并行组用于提升效率，因为Attention层前TP和CP并行组都会调用all-gather，所以可以合并 """
    ... # 全局变量获取
    for ranks in generator_wrapper('tp-cp'):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options('tp_cp', nccl_comm_cfgs))
        if rank in ranks:
           # do samething
  
    ... """ EP，TP+EP等等 """ 

    _set_global_memory_buffer()
```

### Megatron 如何分组？—— 以 TP=2, PP=2, DP=2 为例（共 8 GPU）

#### 📌 步骤 1：按 PP 划分 —— 2 个流水 stage

* Stage 0: GPU [0, 1, 2, 3]
* Stage 1: GPU [4, 5, 6, 7]

#### 📌 步骤 2：在每个 stage 内，按 DP 划分 —— 2 个数据并行组

* Stage 0:
  * DP Group 0: [0, 2]
  * DP Group 1: [1, 3]
* Stage 1:
  * DP Group 0: [4, 6]
  * DP Group 1: [5, 7]

#### 📌 步骤 3：在每个 DP 组内，按 TP 划分 —— 2 个张量并行组

* Stage 0, DP Group 0: [0, 2] → TP 组就是 [0, 2]（TP=2，刚好2个）
* Stage 0, DP Group 1: [1, 3] → TP 组是 [1, 3]
* Stage 1, DP Group 0: [4, 6] → TP 组 [4, 6]
* Stage 1, DP Group 1: [5, 7] → TP 组 [5, 7]

## data\_iterator创建

在当前绝大多数场景下，我们都是用的gpt模型进行训练，对应的megatron也有`GPTDataset`，我们在启动训练前就会创建出`data_iterator`和`Dataloader`，接下来我们看看megatron是如何创建这些关键组件的

```python
def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset   """ 使用的数据集类型 """
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if args.is_instruction_dataset:
        ...  """ 很少用 """
    else:
        """ 关键函数 """
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds
```

上面的代码进入到了BlendedMegatronDatasetBuilder内进行数据集创建，这内部的代码十分冗长，逻辑比较复杂，故放到进阶内容的7.1节进行详细讲述。接下来粗略讲述一下data-iterator的创建

```python
def pretrain(...):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(...) 
    ...
    # Model, optimizer, and learning rate.
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
    ...
    # Data stuff.
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        """ VPP未启用时的逻辑，我们大多数时候都走这里 """
        train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
    print_datetime('after dataloaders are built')


def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    args = get_args()

    """ 使用我们这一章节最开始给出的那个dataset provider函数创建数据集和Dataloader """
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic', 'external']

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None
    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
    else:
        valid_data_iterator = None
    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None
    """ 这个iterator就是将Dataloader包装为可迭代的对象，以方便在gpt_forward_step中使用next(data_iterator)获得数据 """ 
    return train_data_iterator, valid_data_iterator, test_data_iterator
```

从上面可以看到创建的过程，其中Dataloader的创建我这里省略，因为这里读者们自己可以去阅读一下代码，其实已经很容易就能够串起开始的dataset创建和上面给出的iterator创建的过程。

在Dataloader创建的函数中会先创建出三个dataset【train，valid，test】，然后根据三个dataset各自创建出对应的Dataloader【这里的Dataloader就是pytorch的Dataloader】。

有了dataloader之后我们就可以在训练中进行batch数据读取了，这里的逻辑笔者放在下一章的预训练中讲述。

# 预训练：

## 基于iteration的训练：

与传统的机器学习基于epoch来训练的模式不同，LLM内没有了epoch的概念，只有`iteration`的概念——即这次训练会进行N\*M个`forward-backward`操作，N个`step`操作。（`N = iteration`，`M = global_batch_size / micro_batch_size`）

下面我们从代码上看看基于iteration的训练代码：

```python
def train(forward_step_func, model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator, process_non_loss_data_func, config):
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    ... # Setup some training config params
  
    """ 最重要的while训练，里面执行iteration次train_step函数 """
    num_microbatches = get_num_microbatches()
    while iteration < args.train_iters:
        ... """ 更新num microbatch大小 """

        args.curr_iteration = iteration
        """ 调用train_step训练一个step【fwd、bwd、optimizer.step】 """
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        iteration += 1
        batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
        args.consumed_train_samples += batch_size
  
        ... """ logging相关代码 """

        ... """ Evaluation """
  
        ... """ Checkpointing 保存权重文件 """ 
  
    ... """ 所有iteration完成之后的退出逻辑 """

    return iteration, num_floating_point_operations_so_far
```


## train\_step函数：

在megatron中，`forward`和`backward`因为流水线并行而需要定制，所以我们不能像原始的机器学习一样直接把input放入model，然后调用`loss.backward()`计算梯度。原因就是每张卡里的model其实都不一样，对于输入的shape的要求也不一样，

所以我们需要包装`forward`的input，有可能这个input是最开始的输入，也有可能是上一个pipeline stage的output。接下来我们看看对于流水线并行来说`forward`和`backward`的流程：

```python
def train_step(...):
    ...
    """ 清除grad """
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()
    ...
  
    """ 执行前向、反向计算 """
    forward_backward_func = get_forward_backward_func()  """ 获得forward和backward函数，里面会根据PP的设置返回不同的函数 """
    losses_reduced = forward_backward_func(...) 
    ...
  
    """ 对梯度执行Reduce-Scatter操作 """
    optimizer.reduce_model_grads(args, timers)
    ...
  
    """ 更新该DP域持有的梯度对应的部分参数 """
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()
    ...
  
    """ 对更新后的param执行gather操作 """
    if update_successful:
        optimizer.gather_model_params(args, timers)
    ...
  
    """ 通过scheduler更新学习率 """
    if update_successful:
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1
    ...
```
在将fwd和bwd之前，我们有必要了解一下global_batch_size, micro_batch_size, DP_size和num_micro_batches的关系：
```python
 self.num_micro_batches = (running_global_batch_size // micro_batch_times_data_parallel_size)  # 16 // (2 * 4) = 2
```
我们在训练的脚本里会指定这两个size参数，然后megatron会根据DP的数量计算出来这个num_micro_batches的大小，这个数的用处很大，接下来我们将fwd-bwd的时候就会讲到。
