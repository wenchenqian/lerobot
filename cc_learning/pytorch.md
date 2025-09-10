# Pytorch 分布式训练

## DDP训练API

pytorch  ddp训练

```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    """ Environment variables which need to be set when using c10d's default "env" initialization mode. """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
```


可以看到，例子中训练的部分和单机单卡是十分相似的，只是模型用DDP包了一下，然后`Tensor.to()`数据移动时从直接移动到cuda变为了移动到指定的某个RANK去。【上面的demo在模型较小的时候都是可行的】

但随着LLM的兴起，模型越来越大，单个GPU就很难再放下一个完整的模型权重和优化器状态，所以参数共享的便有了需求。于是为了满足参数服务器的功能，DDP训练也需要做出相应的修改，那么接下来我们就看看其做了什么修改吧！

```
import random
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EMBEDDINGS = 30
EMBEDDING_DIM = 8


class HybridModel(torch.nn.Module):
    r"""The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """
    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.device = device
        self.remote_emb_module = remote_emb_module
        # self.fc = DDP(torch.nn.Linear(16, 8).cuda(device), device_ids=[device])
        self.fc = DDP(torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Linear(16, 4)))  # cpu

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        # return self.fc(emb_lookup.cuda(self.device))
        return self.fc(emb_lookup)  # cpu


def _run_trainer(remote_emb_module, rank):
    """
    Each trainer runs a forward pass which involves an embedding lookup on the parameter server and running nn.Linear locally. During the backward pass,
    DDP is responsible for aggregating the gradients for the dense part (nn.Linear) and distributed autograd ensures gradients updates are propagated to the parameter server.
    """
    # Setup the model.
    model = HybridModel(remote_emb_module, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.

    # Retrieve parameters for embedding table and model.fc's parameter.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch(rank):
        for _ in range(1):
            num_indices = random.randint(8, 20)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 5)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            # target = torch.LongTensor(batch_size).random_(4).cuda(rank)
            target = torch.LongTensor(batch_size).random_(4)  # cpu
            yield indices, offsets_tensor, target

    # Train for 10 epochs
    for epoch in range(1, 11):
        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Run distributed optimizer
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different distributed autograd context which hosts different grads


def run_worker(rank, world_size):
    """A wrapper function that initializes RPC, calls the function, and shuts down RPC."""

    # We need to use different port numbers in TCP init_method for init_rpc and init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 6 is master, 7 is ps and 0~5 are trainers.
    if rank == 6:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in range(0, 6):
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank))
            futs.append(fut)

        # Wait for all training to finish.
        for fut in futs:
            fut.wait()
    elif rank <= 5:
        # Initialize process group for Distributed DataParallel on trainers. ---NPU use hccl
        dist.init_process_group(backend="gloo", rank=rank, world_size=6, init_method="tcp://localhost:29500")

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Trainer just waits for RPCs from master.
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 8  # 6 trainers, 1 parameter server, 1 master.
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
```
