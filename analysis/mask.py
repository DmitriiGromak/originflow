import numpy as np
import matplotlib.pyplot as plt
def generate_motif_mask(length, min_motif_length=5, max_motif_length=20):
    """
    生成一个模拟蛋白质结构中motif的mask，其中motif的数量、位置和长度都是随机的。

    Args:
        length (int): 蛋白质结构的总长度。
        min_motif_length (int): 单个motif的最小长度。
        max_motif_length (int): 单个motif的最大长度。

    Returns:
        np.ndarray: 表示motif位置的mask，motif位置为1，非motif位置为0。
    """
    # 根据长度决定motif的数量，这里假设每50个氨基酸有一个motif
    num_motifs = max(1, length // 50)

    # 初始化mask
    mask = np.zeros(length)

    for _ in range(num_motifs):
        # 随机决定每个motif的长度
        motif_length = np.random.randint(min_motif_length, max_motif_length+1)

        # 确保motif不会超出蛋白质结构的长度
        max_start_position = length - motif_length
        if max_start_position <= 0:
            continue  # 如果蛋白质长度不足以容纳一个motif，跳过

        # 随机决定motif的起始位置
        start_position = np.random.randint(0, max_start_position)

        # 在mask中标记motif的位置
        mask[start_position:start_position+motif_length] = 1

    return mask


import torch


def design_masks(B, L, min_motif_len=1, max_motif_len=8,patch=128):
    masks = torch.zeros(B, L)  # 初始化mask张量
    # print(B,L)
    for i in range(B):
        num_motifs = max(1, L // patch)  # 根据L决定motif的数量
        for _ in range(num_motifs):
            # 确保在调用torch.randint前，to参数大于from参数
            motif_len_range = min(max_motif_len, L) + 1 - min_motif_len
            if motif_len_range > 0:
                motif_len = torch.randint(min_motif_len, min(max_motif_len, L) + 1, (1,)).item()
                start_range = L - motif_len + 1
                if start_range > 0:
                    start_pos = torch.randint(0, start_range, (1,)).item()
                    masks[i, start_pos:start_pos + motif_len] = 1
                else:
                    print("Error: start_range is not positive.")
            else:
                #print(max_motif_len,min_motif_len,motif_len_range,num_motifs,L)
                print("Error: motif_len_range is not positive.")

    return masks

def create_binder_mask(batch_com_idx, batch_chain_idx):
    '''
    diffusion area binder is mask=0,fixed area binder is mask=1
    '''
    B, N = batch_com_idx.shape

    # 检测每个样本的com_idx中是否只有一种值
    unique_com_idx = torch.unique(batch_com_idx, dim=1)
    single_value_mask = (unique_com_idx.size(1) == 1)
    if not single_value_mask:
        # 为com_idx创建一个全1的mask，稍后更新
        mask = torch.ones_like(batch_com_idx)

        # 对于com_idx中有多个值的情况，随机挑选一个值并更新mask
        # 使用广播机制，随机挑选每个batch中的一个值
        # 生成每行的随机索引
        rows, cols = unique_com_idx.shape
        random_indices = torch.randint(0, cols, (rows,)).to(batch_com_idx.device)

        # 使用torch.gather选择每行中的随机元素
        selected_values = torch.gather(unique_com_idx, 1, random_indices.unsqueeze(1)).squeeze(1)
        #random_com_values = batch_com_idx[torch.arange(B), selected_values]

        # 将com_idx中等于随机选中值的位置设置为0
        mask[batch_com_idx == selected_values.unsqueeze(1)] = 0

    # 对于com_idx中只有一个值的情况，处理chain_idx
    else :
        # 为com_idx创建一个全1的mask，稍后更新
        mask = torch.ones_like(batch_chain_idx)
        unique_chain_idx = torch.unique(batch_chain_idx, dim=1)
        rows, cols = unique_chain_idx.shape
        random_indices = torch.randint(0, cols, (rows,)).to(unique_chain_idx.device)

        # 使用torch.gather选择每行中的随机元素
        selected_values = torch.gather(unique_chain_idx, 1, random_indices.unsqueeze(1)).squeeze(1)
        # random_com_values = batch_com_idx[torch.arange(B), selected_values]

        # 将com_idx中等于随机选中值的位置设置为0
        mask[batch_chain_idx == selected_values.unsqueeze(1)] = 0

    return mask
# # 示例：生成5个蛋白质长度为200的mask
# B = 5
# L = 200
# mask = design_masks(B, L)
# # 示例：生成一个长度为200的蛋白质结构的mask
# # length = 200
# # mask = generate_motif_mask(length)
#
# # 打印或使用mask
# mask=mask.numpy()
# print(mask)


def plot_masks(masks):
    """
    绘制蛋白质结构的mask。

    Args:
    - masks (torch.Tensor): 表示B个蛋白质结构的mask张量，形状为[B, L]。
    """
    # 确保masks是numpy数组以便绘图
    if isinstance(masks, torch.Tensor):
        masks = masks.numpy()

    # 绘制每个mask
    fig, axs = plt.subplots(masks.shape[0], 1, figsize=(10, masks.shape[0] * 2), sharex=True)
    if masks.shape[0] == 1:
        axs = [axs]  # 如果只有一个mask，确保axs是可迭代的
    for i, ax in enumerate(axs):
        ax.imshow(masks[i:i + 1], aspect="auto", cmap="gray")
        ax.set_ylabel(f"Mask {i + 1}")
        ax.grid(False)
        ax.set_yticks([])
    plt.xticks(range(masks.shape[1]), rotation=90)
    plt.xlabel("Position")
    plt.tight_layout()
    plt.show()


# # 可视化mask（可选）

# plt.figure(figsize=(10, 1))
# plt.imshow(mask[np.newaxis, :], cmap='Greys', aspect='auto')
# plt.yticks([])
# plt.xticks(range(length))
# plt.show()
