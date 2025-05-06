import torch
import numpy as np
from data.so3_utils import angle_from_rotmat,skew_matrix_to_vector
# 计算旋转矩阵之间的 Frobenius 范数
def frobenius_loss(pred, target):
    return torch.norm(pred - target, p='fro', dim=(-2, -1))

# 计算旋转向量之间的损失
def rotation_vector_loss(pred, target):
    pred_rotvec = rotmat_to_rotvec(pred)
    target_rotvec = rotmat_to_rotvec(target)
    return torch.norm(pred_rotvec - target_rotvec, dim=-1)

# 计算四元数之间的损失
def quaternion_loss(pred, target):
    pred_quat = rotmat_to_quaternion(pred)
    target_quat = rotmat_to_quaternion(target)
    return torch.norm(pred_quat - target_quat, dim=-1)

# 将旋转矩阵转换为旋转向量
def rotmat_to_rotvec(rotation_matrices):
    angles, angles_sin, _ = angle_from_rotmat(rotation_matrices)
    vector = skew_matrix_to_vector(rotation_matrices - rotation_matrices.transpose(-2, -1))
    mask_zero = torch.isclose(angles, torch.zeros_like(angles)).to(angles.dtype)
    mask_pi = torch.isclose(angles, torch.full_like(angles, np.pi), atol=1e-2).to(angles.dtype)
    mask_else = (1 - mask_zero) * (1 - mask_pi)
    numerator = mask_zero / 2.0 + angles * mask_else
    denominator = (1.0 - angles**2 / 6.0) * mask_zero + 2.0 * angles_sin * mask_else + mask_pi
    prefactor = numerator / denominator
    vector = vector * prefactor[..., None]
    id3 = torch.eye(3, device=rotation_matrices.device, dtype=rotation_matrices.dtype)
    skew_outer = (id3 + rotation_matrices) / 2.0
    skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3
    vector_pi = torch.sqrt(torch.diagonal(skew_outer, dim1=-2, dim2=-1))
    signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
    signs_line = torch.take_along_dim(skew_outer, dim=-2, indices=signs_line_idx[..., None, None]).squeeze(-2)
    signs = torch.sign(signs_line)
    vector_pi = vector_pi * angles[..., None] * signs
    vector = vector + vector_pi * mask_pi[..., None]
    return vector

# 将旋转矩阵转换为四元数
def rotmat_to_quaternion(rotation_matrices):
    N, L, _, _ = rotation_matrices.shape
    m = rotation_matrices.view(N * L, -1)
    w = torch.sqrt(1.0 + m[:, 0] + m[:, 4] + m[:, 8]) / 2.0
    w4 = 4.0 * w
    x = (m[:, 7] - m[:, 5]) / w4
    y = (m[:, 2] - m[:, 6]) / w4
    z = (m[:, 3] - m[:, 1]) / w4
    quaternions = torch.stack((w, x, y, z), dim=1)
    return quaternions.view(N, L, 4)


# 生成随机旋转矩阵
def generate_random_rotation_matrix():
    random_quaternion = torch.randn(4)
    random_quaternion = random_quaternion / torch.norm(random_quaternion)
    r, i, j, k = random_quaternion
    rot_matrix = torch.tensor([
        [1 - 2*(j**2 + k**2), 2*(i*j - k*r), 2*(i*k + j*r)],
        [2*(i*j + k*r), 1 - 2*(i**2 + k**2), 2*(j*k - i*r)],
        [2*(i*k - j*r), 2*(j*k + i*r), 1 - 2*(i**2 + j**2)]
    ])
    return rot_matrix
# 示例测试函数
def test_rotation_loss():
    mat_t = generate_random_rotation_matrix().unsqueeze(0).unsqueeze(0)*0
    mat_1 = generate_random_rotation_matrix().unsqueeze(0).unsqueeze(0)*0
    frob_loss = frobenius_loss(mat_t, mat_1)
    rotvec_loss = rotation_vector_loss(mat_t, mat_1)
    quat_loss = quaternion_loss(mat_t, mat_1)
    print(f"Frobenius Loss: {frob_loss.item()}")
    print(f"Rotation Vector Loss: {rotvec_loss.item()}")
    print(f"Quaternion Loss: {quat_loss.item()}")

# 执行测试
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from openfold.utils import rigid_utils as ru
from typing import Callable, Dict, Optional, Tuple

Rigid = ru.Rigid
Rotation = ru.Rotation
from chroma.layers.structure.backbone import FrameBuilder

# 示例 Rigid 类
fb = FrameBuilder()


def compute_fape(
        pred_frames: Rigid,
        target_frames: Rigid,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        positions_mask: torch.Tensor,
        length_scale: float,
        l1_clamp_distance: Optional[float] = None,
        eps: float = 1e-8,
) -> torch.Tensor:
    local_pred_pos = pred_frames.invert()[..., None].apply(pred_positions[..., None, :, :])
    local_target_pos = target_frames.invert()[..., None].apply(target_positions[..., None, :, :])

    error_dist = torch.sqrt(torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps)

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def fbb_backbone_loss(
        pred_trans: torch.Tensor,
        pred_rot: torch.Tensor,
        gt_trans: torch.Tensor,
        gt_rot: torch.Tensor,
        mask: torch.Tensor,
        clamp_distance: float = 10.0,
        loss_unit_distance: float = 10.0,
        eps: float = 1e-4,
) -> torch.Tensor:
    pred_aff = Rigid(Rotation(rot_mats=pred_rot), pred_trans)
    gt_aff = Rigid(Rotation(rot_mats=gt_rot), gt_trans)



    fape_loss = compute_fape(
        pred_aff,
        gt_aff,
        mask,
        pred_aff.get_trans(),
        gt_aff.get_trans(),
        mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    return fape_loss


def generate_random_rotation_matrix():
    # 随机生成旋转轴和角度
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)
    angle = torch.randn(1) * torch.pi

    # 使用 Rodrigues' 公式生成旋转矩阵
    K = vector_to_skew_matrix(axis[None, :])[0]
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
    return R


def vector_to_skew_matrix(vectors: torch.Tensor) -> torch.Tensor:
    skew_matrices = torch.zeros((*vectors.shape, 3), device=vectors.device, dtype=vectors.dtype)
    skew_matrices[..., 2, 1] = vectors[..., 0]
    skew_matrices[..., 0, 2] = vectors[..., 1]
    skew_matrices[..., 1, 0] = vectors[..., 2]
    skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)
    return skew_matrices


# if __name__ == '__main__':
#     batch_size = 2
#     num_frames = 100
#     num_atoms = 4
#
#     # 生成两个随机旋转矩阵和平移向量
#     gt_rotmats_1 = generate_random_rotation_matrix().unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, -1, -1)
#     pred_rotmats_1 = generate_random_rotation_matrix().unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, -1, -1)
#
#     gt_trans_1 = torch.randn(batch_size, num_frames, 3)
#     pred_trans_1 = torch.randn(batch_size, num_frames, 3)
#
#     # 生成链索引
#     chain = torch.randint(0, 2, (batch_size, num_frames))
#
#     # 存储误差
#     errors = []
#
#     # 生成一系列逐渐相似的旋转矩阵和平移向量
#     num_steps = 100
#     for i in range(1, num_steps + 1):
#         # 混合旋转矩阵和平移向量，使它们越来越相似
#         alpha = i / num_steps
#         mixed_rotmats_1 = (1 - alpha) * gt_rotmats_1 + alpha * pred_rotmats_1
#         mixed_trans_1 = (1 - alpha) * gt_trans_1 + alpha * pred_trans_1
#
#         # 重新正交化旋转矩阵
#         U, _, V = torch.svd(mixed_rotmats_1)
#         mixed_rotmats_1 = torch.matmul(U, V.transpose(-1, -2))
#
#         # 计算 FAPE 损失
#         loss = fbb_backbone_loss(
#             mixed_trans_1,
#             mixed_rotmats_1,
#             gt_trans_1,
#             gt_rotmats_1,
#             chain
#         )
#         errors.append(torch.mean(loss) / alpha)
#         print(f'Loss at step {i}: {torch.mean(loss) / alpha}')
#
#     # 绘制误差变化图
#     plt.plot(errors)
#     plt.xlabel('Steps')
#     plt.ylabel('FAPE Loss')
#     plt.title('Change in FAPE Loss with Increasingly Similar Rotation Matrices and Translations')
#     plt.show()

