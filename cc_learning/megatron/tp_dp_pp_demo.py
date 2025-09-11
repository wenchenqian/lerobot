# save as: megatron_group_demo.py

def build_megatron_groups(world_size, tp_size, pp_size, dp_size=None, cp_size=None):
    """
    模拟 Megatron-LM 的设备分组逻辑：TP -> DP/CP -> PP
    """
    assert world_size == tp_size * (dp_size or cp_size) * pp_size, "并行配置不匹配总设备数"

    num_dp_cp = dp_size or cp_size
    devices = list(range(world_size))
    groups = {
        'tp': [],
        'dp' if dp_size else 'cp': [],
        'pp': []
    }

    # Step 1: 按 PP 划分
    pp_groups = [devices[i::pp_size] for i in range(pp_size)]
    # 等价于：pp_groups = [devices[i * (world_size//pp_size) : (i+1) * (world_size//pp_size)] for i in range(pp_size)]

    for pp_group in pp_groups:
        groups['pp'].append(pp_group)

        # Step 2: 在每个 PP 组内，按 DP/CP 划分
        dp_cp_groups = [pp_group[i::num_dp_cp] for i in range(num_dp_cp)]

        for dp_cp_group in dp_cp_groups:
            groups['dp' if dp_size else 'cp'].append(dp_cp_group)

            # Step 3: 在每个 DP/CP 组内，按 TP 划分
            tp_group = dp_cp_group  # 因为 TP 是最内层，且 tp_size == len(dp_cp_group)
            groups['tp'].append(tp_group)

    return groups

def print_groups(groups):
    for key, group_list in groups.items():
        print(f"\n🔹 {key.upper()} Groups:")
        for i, g in enumerate(group_list):
            print(f"  Group {i}: {g}")

if __name__ == "__main__":
    # ========== 配置区 ==========
    WORLD_SIZE = 8
    TP = 2
    PP = 2
    DP = 2  # 或设为 CP=2，效果一样

    # ========== 执行分组 ==========
    groups = build_megatron_groups(WORLD_SIZE, TP, PP, dp_size=DP)
    print_groups(groups)

    # ========== 换成 CP 试试 ==========
    print("\n" + "="*50)
    print("换成 CP=2 的情况：")
    groups_cp = build_megatron_groups(WORLD_SIZE, TP, PP, cp_size=2)
    print_groups(groups_cp)