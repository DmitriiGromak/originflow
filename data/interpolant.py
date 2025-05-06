import torch

import numpy as np
from tqdm import tqdm

from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
import copy
from analysis.mask import design_masks,create_binder_mask
from scipy.optimize import linear_sum_assignment
from chroma.data.protein import Protein
from models.symmetry import SymGen
from chroma.layers.structure.mvn import BackboneMVNResidueGas

from chroma.layers.structure.backbone import FrameBuilder
from models.noise_schedule import OTNoiseSchedule
from experiments.utils import get_global_seed

def SS_mask(ss, fixed_mask,design=False):
    if  design:
        return ss.clone() *fixed_mask
    else:
        th = 0.5
    # 计算mask的全局比例
    mask_percentage = th

    # 创建一个与fixed_mask形状相同的随机矩阵
    random_matrix = torch.rand_like(fixed_mask.float())

    # 计算需要被mask的阈值
    threshold = torch.quantile(random_matrix[fixed_mask == 0], mask_percentage)

    # 创建最终的mask，对fixed_mask为0且随机值小于阈值的位置进行mask
    final_mask = (random_matrix < threshold) & (fixed_mask == 0)

    # 将ss中对应final_mask为True的位置设置为0
    new_ss = ss.clone()  # 复制ss避免修改原始数据
    new_ss[final_mask] = 0

    return new_ss
def update_connected_regions_batch(tensor):
    # 初始化结果张量，与输入张量相同的形状
    B, N = tensor.shape
    result = torch.zeros_like(tensor)

    for b in range(B):  # 遍历每个批次
        region_count = 0  # 重置连通区域计数器
        prev_val = 0  # 重置上一个值
        for n in range(N):  # 遍历序列中的每个元素
            # 如果当前值为1且上一个值为0，那么我们遇到了一个新的连通区域
            if tensor[b, n] == 1 and prev_val == 0:
                region_count += 1
            # 更新结果张量的值
            result[b, n] = region_count if tensor[b, n] == 1 else 0
            # 更新上一个值
            prev_val = tensor[b, n]

    return result

def generate_batch_constrained_points_torch(batch_size, n_points, device,base_distance=1.27, constraint_factor=0.0028):
    # Initialize all points at the origin for all batches
    points = torch.zeros(batch_size, n_points, 3,device=device)

    # Generate a random direction for each point in each batch
    random_directions = torch.randn(batch_size, n_points, 3,device=device)
    norms = torch.norm(random_directions, dim=2, keepdim=True)
    random_directions /= norms

    # Apply constraints and calculate the points iteratively (due to dependency on previous point)
    for i in range(1, n_points):
        constraints = -points[:, i-1, :] * constraint_factor * (i ** -0.1)
        adjusted_directions = random_directions[:, i, :] + constraints
        adjusted_directions /= torch.norm(adjusted_directions, dim=1, keepdim=True)

        points[:, i, :] = points[:, i-1, :] + adjusted_directions * base_distance
    points=torch.tensor(points,device=device)
    points=points - torch.mean(points, dim=-2, keepdims=True)
    return points
def _centered_gaussian(num_batch, num_res, atoms,device='cuda'):
    if atoms is not None:
        noise = torch.randn(num_batch, num_res,atoms, 3, device=device)
        return noise - torch.mean(noise, dim=[1,2], keepdims=True)
    else:
        noise = torch.randn(num_batch, num_res, 3, device=device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


def save_pdb_chain(xyz,chain_idx, pdb_out="out.pdb"):


    chains='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    ATOMS = ["N","CA","C","O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%4],"GLY",chains[int(chain_idx[a]-1)],a+1,x,y,z,1,0)
        )
        k += 1
        if k % 4 == 0: a += 1
    out.close()
def save_pdb(xyz, pdb_out="out.pdb"):
    pdb_out=pdb_out
    ATOMS = ["N","CA","C","O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%4],"GLY","A",a+1,x,y,z,1,0)
        )
        k += 1
        if k % 4 == 0: a += 1
    out.close()




def invert_noise_via_optimization(model, batch_copy, num_opt_steps=500, lr=1e-3):
    """
    反解高斯噪声: 给定目标真实坐标，反向优化获得初始高斯噪声。
    只对fixed_mask=1的部分进行优化，保持原始batch结构。
    """
    model.eval()
    

    
    # 获取fixed_mask和设备
    fixed_mask = batch_copy['fixed_mask']
    device = fixed_mask.device
    
    # 获取真实目标坐标
    trans_true = batch_copy['trans_1']
    rotmats_true = batch_copy['rotmats_1']
    
    # 创建完整尺寸的噪声张量
    trans_noise = batch_copy['trans_t']
    rotmats_noise = batch_copy['rotmats_t']
    
    # 设置时间t=0
    t_0 = torch.zeros_like(batch_copy['t'] if 't' in batch_copy else torch.zeros((trans_true.shape[0], 1), device=device))
    batch_copy['t'] = t_0
    
    # 将噪声张量中fixed_mask=1的部分设为可训练参数
    # 为此我们创建一个噪声"容器"并通过.data来更新它
    trans_noise.requires_grad_(True)
    rotmats_noise.requires_grad_(True)
    
    # 只优化fixed_mask=1部分的参数
    optimizer = torch.optim.Adam([
        {'params': [trans_noise], 'lr': lr},
        {'params': [rotmats_noise], 'lr': lr}
    ])
    
    for step in tqdm(range(num_opt_steps), desc="Optimizing noise"):
        optimizer.zero_grad()
        
        # 更新batch中的噪声
        batch_copy['trans_t'] = trans_noise
        batch_copy['rotmats_t'] = rotmats_noise
        
        # 前向传播
        model_out = model(batch_copy, recycle=1, is_training=True)
        
        # 只计算fixed_mask=1部分的损失
        pred_trans = model_out['pred_trans']
        pred_rotmats = model_out['pred_rotmats']
        
        # 创建只包含fixed_mask=1部分的损失掩码
        loss_mask_trans = fixed_mask.unsqueeze(-1).expand_as(trans_true)
        loss_mask_rotmats = fixed_mask.unsqueeze(-1).unsqueeze(-1).expand_as(rotmats_true)
        
        # 计算有掩码的损失
        loss_trans = (((pred_trans - trans_true) ** 2) * loss_mask_trans.float()).sum() / loss_mask_trans.float().sum()
        loss_rot = (((pred_rotmats - rotmats_true) ** 2) * loss_mask_rotmats.float()).sum() / loss_mask_rotmats.float().sum()
        loss = loss_trans + loss_rot
        
        # 反向传播
        loss.backward()
        
        # 手动将梯度置零（对于fixed_mask=0的部分）
        with torch.no_grad():
            trans_noise.grad.masked_fill_(~loss_mask_trans, 0)
            rotmats_noise.grad.masked_fill_(~loss_mask_rotmats, 0)
        
        # 更新参数
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}, loss: {loss.item():.5f}, trans_loss: {loss_trans.item():.5f}, rot_loss: {loss_rot.item():.5f}")
    
    # 返回优化后的噪声
    return trans_noise.detach(), rotmats_noise.detach()

class Interpolant_10:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

        self.backbone_init = BackboneMVNResidueGas().cuda()
        # Frame trunk
        self.frame_builder = FrameBuilder()

        # Noise schedule
        self.noise_perturb=OTNoiseSchedule()



        self._eps=1e-6
    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_bbatoms(self, trans_1, chain_idx, t, res_mask):
        """
        corrupt backbone atoms with a batchot, corrupt them Ca, and others respectively
        so other backbone atoms is not so far from Ca, to elucate the influence of searching space
        Args:trans_1:data  [B,N,4,3],trans_0:0

        Notes: 1-t used for gt, which is same direction with my paper


        """
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # 1. sample Ca and use batchot to solve it to get trans_t
        trans_nm_0 = _centered_gaussian(*res_mask.shape, None, self._device)
        trans_0 = trans_nm_0 * rg
        trans_1_ca = trans_1[..., 1, :]
        trans_0 = self._batch_ot(trans_0, trans_1_ca, res_mask)
        # trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1[..., 1, :]

        trans_z = trans_0 / rg  # nm _SCALE

        # 2. sample N C O and use batchot to solve it to get others_t
        num_batch = chain_idx.shape[0]
        num_residues = chain_idx.shape[1]

        # noisy_rotmats = self.igso3.sample(
        #     torch.tensor([1.5]),
        #     num_batch * num_residues
        # ).to(self._device)
        # noisy_rotmats = noisy_rotmats.reshape(num_batch, num_residues, 3, 3)

        #so3z=self.frame_builder(noisy_rotmats, torch.zeros_like(trans_0), chain_idx)


        z = torch.rand(num_batch, num_residues, 4, 3).to(trans_1.device)
        mask = torch.zeros(num_batch, num_residues, 4, ).to(trans_1.device)
        mask[..., 1] = 1
        others_z = z * (1 - mask[..., None])  ##nm _SCALE

        # others_1 = trans_1 - trans_1[..., 1, :].unsqueeze(-2).repeat(1, 1, 4, 1)  #ANG_SCALE
        # others_1 = others_1/ du.NM_TO_ANG_SCALE  # nm_SCALE
        # others_t = (1 - t[..., None]) * others_z + t[..., None] * others_1  # nm_SCALE

        # 3. combine and transform to resgas
        self.backbone_init._register(stddev_CA=rg, device=self._device)
        z = mask[..., None] * trans_z.unsqueeze(-2).repeat(1, 1, 4, 1) + others_z
        bbatoms = self.backbone_init.sample(chain_idx, Z=z)
        xt = (1 - t[..., None, None]) *bbatoms   + t[..., None, None] *  trans_1  # nm_SCALE

        # forward in z space
        # z_0=self.backbone_init._multiply_R_inverse(trans_1,chain_idx)
        # z_hat=(1 - t[..., None, None]) * z_0 + t[..., None, None] * z  # nm_SCALE
        # xt_hat=self.backbone_init.sample( chain_idx,Z=z_hat)
        # print(torch.mean(xt_hat-xt))

        return xt * res_mask[..., None, None]

    # def _corrupt_trans(self, trans_1, t,rg, res_mask):
    #     trans_nm_0 = _centered_gaussian(*res_mask.shape, None, self._device)
    #     trans_0 = trans_nm_0 * rg
    #     trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
    #     trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
    #     trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
    #     return trans_t * res_mask[..., None]

    def _corrupt_trans(self, trans_1, t,rg,  res_mask,fixed_mask=None,design=False):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, None, self._device)
        trans_0 = trans_nm_0 * rg
        if not design:

            trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        else:
            trans_0 = trans_0
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1

        if fixed_mask is not None:
            diffuse_mask=res_mask * (1-fixed_mask)
            trans_t = _trans_diffuse_mask(trans_t, trans_1,diffuse_mask )
        else:
            trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]


    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask,fixed_mask=None):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch * num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + identity[None, None] * (1 - res_mask[..., None, None])
        )

        if fixed_mask is not None:
            diffuse_mask=res_mask * (1-fixed_mask)
            rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1,diffuse_mask )
        else:
            rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

        return rotmats_t

    def _center(self,batch):
        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center_before.pdb')
        if 'bbatoms' not in batch and 'atoms14' in batch:
            bb_pos = batch['atoms14'][:, :, 1]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'], dim=1) + 1e-5)[:, None]
            batch['atoms14'] = batch['atoms14'] - bb_center[:, None, None, :]

        else:
            bb_pos = batch['bbatoms'][:,:, 1]*batch['res_mask'][...,None]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'],dim=1) + 1e-5)[:,None]
            batch['bbatoms'] = batch['bbatoms'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch

    def _center_bbatoms_t(self,batch):
        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center_before.pdb')

        bb_pos = batch['bbatoms_t'][:,:, 1]*batch['res_mask'][...,None]
        bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'],dim=1) + 1e-5)[:,None]
        batch['bbatoms_t'] = batch['bbatoms_t'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch
    def _motif_center(self,batch):
        '''
        only use the center of the fixed area, 这样尝试避免复合物距离过大
        '''
        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center_before.pdb')
        if 'bbatoms' not in batch and 'atoms14' in batch:
            bb_pos = batch['atoms14'][:, :, 1]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['fixed_mask'], dim=1) + 1e-5)[:, None]
            batch['atoms14'] = batch['atoms14'] - bb_center[:, None, None, :]

        else:
            bb_pos = batch['bbatoms'][:,:, 1]*(batch['fixed_mask'])[...,None]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum((batch['fixed_mask']),dim=1) + 1e-5)[:,None]
            batch['bbatoms'] = batch['bbatoms'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch

    def _binder_center(self,batch):
        '''
        only use the center of the fixed area, 这样尝试避免复合物距离过大
        '''

        bb_pos = batch['bbatoms'][:,:, 1]*(batch['fixed_mask'])[...,None]
        bb_center = torch.sum(bb_pos, dim=1) / (torch.sum((batch['fixed_mask']),dim=1) + 1e-5)[:,None]
        batch['bbatoms'] = batch['bbatoms'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch

    def corrupt_batch(self, batch):

        # for rcsb should be centered
        batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)
        #scope
        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        #rcsb
        bbatoms = noisy_batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1
        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        if 'chain_idx'  not in noisy_batch:
            chain_idx = torch.ones_like(res_mask)
            noisy_batch['chain_idx'] = chain_idx
            noisy_batch['bbatoms'] = self.frame_builder(rotmats_1, trans_1, chain_idx).float()

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        noisy_batch['fixed_mask'] = None
        return noisy_batch



    def corrupt_batch_base_ss(self, batch):
        '''

        we add ss info,  and train base model

        '''

        # for rcsb should be centered
        batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)



        #rcsb
        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1
        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        fixed_mask=torch.zeros_like(res_mask)
        batch['ss'] = SS_mask(batch['ss'],fixed_mask,design=False)
        noisy_batch['ss'] = batch['ss']







        if 'chain_idx'  not in noisy_batch:
            chain_idx = torch.ones_like(res_mask)
            noisy_batch['chain_idx'] = chain_idx
            noisy_batch['bbatoms'] = self.frame_builder(rotmats_1, trans_1, chain_idx).float()

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        noisy_batch['fixed_mask'] = fixed_mask
        return noisy_batch


    def corrupt_seq(self, batch,aug_eps=0.1):

        # for rcsb should be centered
        batch['bbatoms']= batch['atoms4']
        batch=self._center(batch)
        batch['bbatoms']= batch['bbatoms']+torch.randn_like(batch['bbatoms'])*aug_eps

        noisy_batch = copy.deepcopy(batch)
        #scope
        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        #rcsb
        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1
        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape


        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = torch.ones_like(t)

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)


        noisy_batch['trans_t'] = trans_1



        noisy_batch['rotmats_t'] = rotmats_1
        noisy_batch['bbatoms_t'] = batch['bbatoms']
        noisy_batch['fixed_mask'] = res_mask
        return noisy_batch
    def fixt_corrupt_batch(self, batch,t):
        # batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)

        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]

        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask)
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        xt=self._corrupt_bbatoms(bbatoms,chain_idx, t, res_mask)
        rotmats_t2, trans_t2, _q = self.frame_builder.inverse(xt, chain_idx)  # frames in new rigid system



        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()

        noisy_batch['trans_t2'] = trans_t2
        noisy_batch['rotmats_t2'] = rotmats_t2
        noisy_batch['bbatoms_t2'] = self.frame_builder(rotmats_t2, trans_t2, chain_idx).float()

        noisy_batch['fixed_mask'] = None
        return noisy_batch
    def corrupt_batch_motif(self, batch):
        '''
        这部分扰动坐标，主要是随机生成一个mask，被mask住的部分其trans和rot 数据保持不动，用于训练motif模型
        '''

        noisy_batch = copy.deepcopy(batch)

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        #make res mask
        motifornot = torch.rand(1, device=self._device)
        #fixed_mask = design_masks(res_mask.shape[0], res_mask.shape[1]).to(self._device)
        #noisy_batch['fixed_mask'] = fixed_mask
        if motifornot > 0.5:
            fixed_mask = design_masks(res_mask.shape[0], res_mask.shape[1]).to(self._device)
            noisy_batch['fixed_mask'] = fixed_mask

            noisy_batch = self._motif_center(noisy_batch)
            # #recenter
            # center_CA = ((fixed_mask[:, :, None]) * bbatoms[..., 1, :]).sum(dim=1) / (
            #         (fixed_mask[:, :, None]).sum(dim=1) + 1e-4)  # (B,  3)
            # bbatoms = bbatoms - center_CA[:, None, None, :]
            # print(noisy_batch['fixed_mask'].sum(dim=-1) / noisy_batch['fixed_mask'].shape[1])
        else:
            fixed_mask = res_mask*0
            noisy_batch['fixed_mask'] = fixed_mask # we do not fixed any pos, model generate all

            noisy_batch=self._center(noisy_batch)


        # sample area set to zero




        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')

        # must after center

        bbatoms = noisy_batch['bbatoms'].float()  # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1



        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask,fixed_mask)
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask,fixed_mask)

        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        # save_pdb(noisy_batch['bbatoms_t'][0].reshape(-1,3), 'motif_test.pdb')
        # save_pdb(bbatoms[0].reshape(-1,3), 'motif_gt.pdb')


        return noisy_batch





    def corrupt_batch_binder_sidechain(self, batch,precision,t=None,noise=False):
        '''
        这部分主要是用于binder的训练，增加了SS 数据，和毗邻矩阵，且也增加了b-factor的部分，还有chi 角
        noise: when train side, could noise binder area or not
        '''

        batch=self._center(batch)
        if precision == 'fp16':
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].half()
        else:
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].float()



        noisy_batch = copy.deepcopy(batch)
        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')
        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        #print(res_mask.shape)
        bbatoms = batch['atoms14'][...,:4,:] # Angstrom

        binder_mask=create_binder_mask(batch['com_idx'],batch['chain_idx']).int()
        noisy_batch['fixed_mask'] = binder_mask
        noisy_batch['bbatoms'] = bbatoms
        fixed_mask=binder_mask


        batch['ss'] = SS_mask(batch['ss'],fixed_mask)


        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1


        if t is None:
            # [B, 1]
            t = self.sample_t(num_batch)[:, None]
            noisy_batch['t'] = t
        else:
            t_ = self.sample_t(num_batch)[:, None]
            t = (t_*(1-t))+t
            noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)



        if not noise:
            noisy_batch['trans_t'] = trans_1
            noisy_batch['rotmats_t'] = rotmats_1
            noisy_batch['bbatoms_t'] =bbatoms.float()

        else:
            trans_t = self._corrupt_trans(trans_1, t, rg, res_mask, fixed_mask)
            rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask, fixed_mask)
            noisy_batch['trans_t'] = trans_t
            noisy_batch['rotmats_t'] = rotmats_t
            noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()


        save_pdb_chain(noisy_batch['bbatoms_t'][0].reshape(-1,3).cpu().numpy(),chain_idx[0].cpu().numpy(), 'binder_1.pdb')


        return noisy_batch


    def corrupt_batch_binder(self, batch,precision,binder_mask=None,design_num=0,t=None,design=False,path=None,hotspot=False):
        '''
        这部分主要是用于binder的训练，增加了SS 数据，和毗邻矩阵，且也增加了b-factor的部分，还有chi 角
        binder mask 是要固定的位置

        hotspot  if true, it as center
        '''

        # save_pdb_chain(batch['atoms14'][0][:,:4,:].reshape(-1, 3).cpu().numpy(), batch['chain_idx'][0].cpu().numpy(),
        #                f'/{path}/native_nonoise_{str(design_num)}.pdb')

        # if not hotspot:
        #     # use new center
        #     if design:
        #         # use fixed area to compute zeros
        #         batch = self._motif_center(batch)
        #     else:
        #         # batch=self._center(batch)  如果对整体进行中心化，其实会暴露binder的位置，所以应该以target为中心
        #         batch = self._motif_center(batch)


        if precision == 'fp16':
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].half()
        else:
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].float()



        noisy_batch = copy.deepcopy(batch)

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        # print(res_mask.shape)
        bbatoms = batch['atoms14'][...,:4,:] # Angstrom

        if binder_mask is None:
            binder_mask=create_binder_mask(batch['com_idx'],batch['chain_idx']).int()
        else:
            binder_mask=binder_mask.int()
        noisy_batch['fixed_mask'] = binder_mask
        noisy_batch['bbatoms'] = bbatoms
        fixed_mask=binder_mask

        if not hotspot:
            noisy_batch = self._binder_center(noisy_batch)

        batch['ss'] = SS_mask(batch['ss'],fixed_mask,design)
        noisy_batch['ss'] = batch['ss']

        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(noisy_batch['bbatoms'] , chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1


        if t is None:
            # [B, 1]
            t = self.sample_t(num_batch)[:, None]
            noisy_batch['t'] = t
        else:
            ## when inference, set t=0,mean start from 1st  t=1 mean fixed not change
            sample_t = self.sample_t(num_batch)[:, None]
            t=torch.ones_like(sample_t)*t
            noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        if not design:
            trans_t = self._corrupt_trans(trans_1, t,rg, res_mask,fixed_mask,design)
            rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask,fixed_mask)
        else:
            trans_0 = _centered_gaussian(
                num_batch, trans_1.shape[1], None, self._device) * rg

            diffuse_mask=res_mask * (1-fixed_mask)
            trans_t = _trans_diffuse_mask(trans_0, trans_1,diffuse_mask )

            rotmats_0 = _uniform_so3(num_batch, trans_1.shape[1], self._device)

            rotmats_t = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)

        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()

        # noisy_batch=self._center_bbatoms_t(noisy_batch)
        # rotmats_t,trans_t, _q=self.frame_builder.inverse(noisy_batch['bbatoms_t'],chain_idx)
        # noisy_batch['trans_t']=trans_t
        # noisy_batch['rotmats_t'] =rotmats_t

        if path is not None:


            save_pdb_chain(noisy_batch['bbatoms_t'][0].reshape(-1,3).cpu().numpy(),chain_idx[0].cpu().numpy(), f'/{path}/center_{str(design_num)}.pdb')

            # save_pdb_chain(bbatoms[0].reshape(-1, 3).cpu().numpy(), chain_idx[0].cpu().numpy(),
            #                f'{path}/native_x0_{str(design_num)}.pdb')

        return noisy_batch

    def corrupt_batch_atoms(self, batch):
        # batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)

        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        xt = self._corrupt_bbatoms(bbatoms, chain_idx, t, res_mask)
        rotmats_t, trans_t, _q = self.frame_builder.inverse(xt, chain_idx)  # frames in new rigid system

        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        noisy_batch['fixed_mask'] = None
        return noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / ( 1- t)
        return trans_t + trans_vf * d_t

    def _X_Temp_euler_step(self, d_t, t, trans_1, trans_t,):
        alpha_t=t
        lambda_0=self._cfg.temp

        if lambda_0 is None:
            lambda_0=1
        temp=lambda_0 / (alpha_t ** 2 + (1 - alpha_t ** 2) * lambda_0)

        h=1/t
        g2=(2-2*t)/t
        score=(t*trans_1-trans_t)/(1-t)**2
        vf=h*trans_t+0.5*g2*score*temp

        if torch.any(torch.isinf(vf)):
            print(vf)
            print(score)
            raise ValueError(' inf in vf')



        #trans_vf = (trans_1 - trans_t) / ( 1- t)

        return trans_t + vf * d_t

    def _X_Temp_motif_euler_step(self, d_t, t, trans_f,trans_1, trans_t,fixed_mask,theta):
        temp = self._cfg.temp
        h=1/t
        g2=(2-2*t)/t
        score_motif=(fixed_mask[...,None]*(t*trans_f-trans_t)/(1-t)**2)*theta
        score=1*((t*trans_1-trans_t)/(1-t)**2)+ score_motif
        vf=h*trans_t+0.5*g2*score*temp


        #trans_vf = (trans_1 - trans_t) / ( 1- t)

        return trans_t + vf * d_t

    def sde(self, d_t, t, trans_1, trans_t,chain_idx,temp=10):
        h=1/t
        g2=(2-2*t)/t
        score=(t*trans_1-trans_t)/(1-t)**2
        vf=(h*trans_t-g2*score)* d_t

        dw=torch.randn_like(trans_1)
        dx=vf  + torch.sqrt(g2)*dw
        #trans_vf = (trans_1 - trans_t) / ( 1- t)

        return trans_t + dx

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)
    def _rots_motif_euler_step(self, d_t, t, rotmats_f,rotmats_1,rotmats_t,fixed_mask,theta):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')

        mid_rot=so3_utils.geodesic_t(
            theta, rotmats_f, rotmats_1)

        mid_rot=rotmats_1* (1 - fixed_mask[...,None,None]) +  mid_rot * fixed_mask[...,None,None]
        return so3_utils.geodesic_t(
            scaling* d_t, mid_rot, rotmats_t)

    def elbo(self, X0_pred, X0, C, t, loss_mask):
        """ITD ELBO as a weighted average of denoising error,
        inspired by https://arxiv.org/abs/2302.03792"""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(X0.device)

        # Interpolate missing data with Brownian Bridge posterior
        # X0 = backbone.impute_masked_X(X0, C)
        # X0_pred = backbone.impute_masked_X(X0_pred, C)

        # Compute whitened residual

        X0 = X0 * loss_mask[..., None, None]
        X0_pred = X0_pred * loss_mask[..., None, None]
        dX = (X0 - X0_pred).reshape([X0.shape[0], -1, 3])
        R_inv_dX = self.backbone_init._multiply_R_inverse(dX, C)

        # Average per atom, including over "missing" positions that we filled in
        weight = 0.5 * self.noise_perturb.SNR_derivative(t)[:, None]
        snr = self.noise_perturb.SNR(t)[:, None]

        c = R_inv_dX.pow(2)
        v = 1 / (1 + snr)

        # Compute per-atom loss

        # loss_itd = (
        #     weight * (R_inv_dX.pow(2) )
        #     - 0.5 * np.log(np.pi * 2.0 * np.e)
        # ).reshape(X0.shape)

        # if minus 1/1+snr  the lossidt could be zheng
        loss_itd = (
                weight * (R_inv_dX.pow(2) - 1 / (1 + snr))
                - 0.5 * np.log(np.pi * 2.0 * np.e)
        ).reshape(X0.shape)

        # Compute average per-atom loss (including over missing regions)
        mask = loss_mask.float()
        mask_atoms = mask.reshape(mask.shape + (1, 1)).expand([-1, -1, 4, 1])

        # Per-complex
        elbo_gap = (mask_atoms * loss_itd).sum([1, 2, 3])
        logdet = self.backbone_init.log_determinant(C)
        elbo_unnormalized = elbo_gap   - logdet

        # Normalize per atom
        elbo = elbo_unnormalized / (mask_atoms.sum([1, 2, 3]) + self._eps)

        # Compute batch average
        weights = mask_atoms.sum([1, 2, 3])
        elbo_batch = (weights * elbo).sum() / (weights.sum() + self._eps)
        mmse= (c).sum() / (weights.sum() + self._eps)
        return elbo, elbo_batch
    def pseudoelbo(self, loss_per_residue, C, t):
        """Compute pseudo-ELBOs as weighted averages of other errors."""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(C.device)
        self._eps=1e-5
        # Average per atom, including over x"missing" positions that we filled in
        weight = 0.5 * self.noise_perturb.SNR_derivative(t)[:, None]
        loss = weight * loss_per_residue

        # Compute average loss
        mask = (C > 0).float()
        pseudoelbo = (mask * loss).sum(-1) / (mask.sum(-1) + self._eps)
        pseudoelbo_batch = (mask * loss).sum() / (mask.sum() + self._eps)
        return pseudoelbo, pseudoelbo_batch
    def _loss_pseudoelbo(self,  X0_pred, X, C, t, w=None, X_t_2=None):
        # Unaligned residual pseudoELBO
        self.loss_scale=10
        unaligned_mse = ((X - X0_pred) / self.loss_scale).square().sum(-1).mean(-1)
        elbo_X, batch_pseudoelbo_X = self.pseudoelbo(
            unaligned_mse, C, t
        )
        return elbo_X,batch_pseudoelbo_X

    def _se3_loss(self,noisy_batch,model_output,training_cfg,_exp_cfg):

        loss_mask = noisy_batch['res_mask']
        num_batch, num_res = loss_mask.shape
        # Ground truth labels
        gt_trans_0 = noisy_batch['trans_1']
        gt_rotmats_0 = noisy_batch['rotmats_1']
        gt_bb_atoms = noisy_batch['bbatoms']
        chain_idx = noisy_batch['chain_idx']

        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t.type(torch.float32), gt_rotmats_0.type(torch.float32))


        # Timestep used for normalization.
        # in frameflow t~1, is same to ~ gt, normscale~0, so normscale ~ 1-t
        # in ours t~0, is same to ~ gt, normscale~0, so normscale=t
        t = noisy_batch['t']
        norm_scale =  torch.max(
            t[..., None], 1-torch.tensor(training_cfg.t_normalize_clip))



        # Model output predictions.
        pred_trans_0 = model_output['pred_trans']
        pred_rotmats_0 = model_output['pred_rotmats']


        # Backbone atom loss
        pred_bb_atoms = self.frame_builder(pred_rotmats_0, pred_trans_0, chain_idx)
        #vio_loss = self._loss_vio(loss_mask, pred_bb_atoms, noisy_batch['aatype'], noisy_batch['res_idx'], norm_scale)

        # Loss=RL( pred_bb_atoms,gt_bb_atoms,noisy_batch['bbatoms_t'],chain_idx,t,loss_mask)
        # elbo


        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom
        bb_atom_loss=bb_atom_loss

        # Translation VF loss
        trans_error = (gt_trans_0 - pred_trans_0) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf ) / norm_scale
        R_ij_error = ((gt_rotmats_0 - pred_rotmats_0).square().sum([-1, -2]) * loss_mask).sum(-1)
        # rots_vf_error =  so3_utils.calc_rot_vf(
        #     gt_rotmats_1.type(torch.float32),pred_rotmats_1.type(torch.float32))/ norm_scale
        if torch.any(torch.isnan(rots_vf_error)):
            print('gt_rotmats_1:', gt_rotmats_0)
            print('pred_rotmats_1:', pred_rotmats_0)
            print(torch.mean(gt_rotmats_0 - pred_rotmats_0))

        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        rots_vf_loss = rots_vf_loss #+ R_ij_error / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 4, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 4, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 4])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 4])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        dist_mat_loss=dist_mat_loss

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss)* (
                t> training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= _exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        se3_vf_loss = se3_vf_loss #+ vio_loss

        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "R_ij_error": R_ij_error / loss_denom,

        }

    def sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)
        res_idx=torch.arange(num_res,device=self._device)[None]
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_idx':res_idx ,
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'fixed_mask':res_mask*0,
            'ss':torch.zeros_like(res_mask).long(),}

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        print(f'Running {self._sample_cfg.num_timesteps} timesteps')
        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        S=torch.ones_like(chain_idx)
        return atom37_traj.detach().cpu(),chain_idx.detach().cpu(),S.detach().cpu()

    def try_motif_sample(
            self,
            num_batch,
            num_res: list,
            model,

            chain_idx,
            native_X,
            mode='motif',
            fixed_mask=None,
            training=False,
    ):

        num_res=num_res[0]
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)
        res_idx=torch.arange(num_res,device=self._device)[None]
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0



        batch = {
            'res_idx':res_idx ,
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'fixed_mask':fixed_mask,
            'ss':torch.zeros_like(res_mask).long(),
            'bbatoms':native_X,}


        motif_X = self._motif_center(batch)['bbatoms']


        rotmats_m,trans_m, _=self.frame_builder.inverse(motif_X, chain_idx)

        # fix motif area
        diffuse_mask=res_mask*(1-fixed_mask)
        trans_0 = _trans_diffuse_mask(trans_0, trans_m, diffuse_mask)
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_m, diffuse_mask)




        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        print(f'Running {self._sample_cfg.num_timesteps} timesteps')
        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        S=torch.ones_like(chain_idx)
        return atom37_traj.detach().cpu(),chain_idx.detach().cpu(),S.detach().cpu()


    def hybrid_sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4

        self.backbone_init._register(stddev_CA=rg, device=res_mask.device)
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'fixed_mask': None,}

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X
            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)


            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2


        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        S=torch.ones_like(chain_idx)
        return atom37_traj.detach().cpu(),chain_idx.detach().cpu(),S.detach().cpu()



    def _init_atoms_backbone(self,num_batch, num_res,chain_idx):

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg
        trans_z = trans_0 / rg  # nm _SCALE

        # 2. sample N C O and use batchot to solve it to get others_t
        num_batch = chain_idx.shape[0]
        num_residues = chain_idx.shape[1]


        z = torch.rand(num_batch, num_residues, 4, 3).to(trans_0.device)
        mask = torch.zeros(num_batch, num_residues, 4, ).to(trans_0.device)
        mask[..., 1] = 1
        others_z = z * (1 - mask[..., None])  ##nm _SCALE

        # 3. combine and transform to resgas
        self.backbone_init._register(stddev_CA=rg, device=self._device)
        z = mask[..., None] * trans_z.unsqueeze(-2).repeat(1, 1, 4, 1) + others_z
        bbatoms = self.backbone_init.sample(chain_idx, Z=z) # nm_SCALE
        return bbatoms

    def _init_complex_backbone(self,num_batch, num_res):

        chain_idx = torch.cat(
            [torch.full([rep], i+1) for i, rep in enumerate(num_res)]
        ).to(self._device).expand(num_batch, -1)
        res_mask = torch.ones_like(chain_idx)

        num_res=chain_idx.shape[1]

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        return res_mask, chain_idx, trans_0, rotmats_0,rg
    def hybrid_Complex_sample(
            self,
            num_batch,
            num_res,

            model,
            ss=None,
    ):

        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        if ss is None:
            ss = res_mask

        batch = {
            'res_mask': res_mask,
            'fixed_mask': torch.zeros_like(res_mask),
            'chain_idx': chain_idx,
            'ss': torch.tensor(ss).to(res_mask.device).unsqueeze(0),
                }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        X=atom37_traj
        C=chain_idx+25
        S=torch.ones_like(C)
        return X,C,S


    def hybrid_Complex_sym_sample(
            self,
            num_batch,
            num_res,
            # ss,
            model,
            symmetry='c4',
            recenter=True,
            radius=0,
    ):

        self.symmetry=SymGen(
                symmetry,
                recenter,
                radius,
            )


        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)

        idx_pdb=torch.arange(num_res[0]).long().to(chain_idx.device).unsqueeze(0).repeat(chain_idx.shape[0], 1)
        idx_pdb, chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        chains=[]
        from data.utils import chain_str_to_int,CHAIN_TO_INT
        for i in chain_idx:
            chains.append(int(CHAIN_TO_INT[i]))


        chain_idx=torch.tensor(chains).to(res_mask.device).unsqueeze(0).repeat(res_mask.shape[0], 1)
        #intialize S
        S = torch.ones_like(chain_idx)


        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_mask': res_mask,
            'fixed_mask': None,
            'chain_idx': chain_idx,
            'ss': torch.zeros_like(res_mask),
                }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take forward step
            bb_atoms_pred = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
            sym_bb_atoms_pred,_ = self.symmetry.apply_symmetry(bb_atoms_pred.squeeze(0).to('cpu'), S.squeeze(0).to('cpu'))
            sym_bb_atoms_pred=sym_bb_atoms_pred.to(res_mask.device)
            pred_rotmats_1, pred_trans_1, q=self.frame_builder.inverse(sym_bb_atoms_pred.unsqueeze(0),chain_idx)

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            # Apply symmetry after denoise
            bb_atoms_pred = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            sym_bb_atoms_pred ,_= self.symmetry.apply_symmetry(bb_atoms_pred.squeeze(0).to('cpu'), S.squeeze(0).to('cpu'))



            sym_bb_atoms_pred = sym_bb_atoms_pred.to(res_mask.device)
            rotmats_t_2, trans_t_2, q=self.frame_builder.inverse(sym_bb_atoms_pred.unsqueeze(0),chain_idx)



            p = Protein.from_XCS(sym_bb_atoms_pred.unsqueeze(0), chain_idx, res_mask, )
            # p.to_PDB('sym_native_test'+str(t_1)+'.pdb')



            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )

        # Apply symmetry after denoise
        bb_atoms_pred = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
        sym_bb_atoms_pred,_ =  self.symmetry.apply_symmetry(bb_atoms_pred.squeeze(0).to('cpu'), S.squeeze(0).to('cpu'))
        sym_bb_atoms_pred = sym_bb_atoms_pred.to(res_mask.device)
        pred_rotmats_1, pred_trans_1, q = self.frame_builder.inverse(sym_bb_atoms_pred.unsqueeze(0),chain_idx)


        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj
        C=chain_idx+25

        return X,C,S


    def hybrid_Complex_sample_bybinder(
            self,
            num_batch,
            num_res,
            model,
    ):

        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)
        length_res=res_mask.shape[1]
        trans_0 = trans_0
        rotmats_0 = rotmats_0

        fixed_mask=torch.zeros_like(res_mask)

        batch = {
            'res_mask': res_mask,
            'fixed_mask': fixed_mask,
            'chain_idx': chain_idx,

            'ss': torch.zeros_like(res_mask).to(torch.long),
            'aatype': torch.ones_like(res_mask).to(torch.long)*20,
            'chi': torch.zeros(size=(num_batch, length_res,4), device=self._device),
            'mask_chi': torch.zeros(size=(num_batch, length_res,4), device=self._device),

            'res_idx': torch.range(0, length_res - 1, device=self._device).unsqueeze(0),

            'atoms14_b_factors': torch.zeros(size=(num_batch, length_res,4), device=self._device),

                }

        # Set-up time
        # ts = torch.linspace(
        #     self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        ts = torch.linspace(
            1e-2, 1.0, 100)


        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in tqdm(ts[1:]):

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch, recycle=1, is_training=True)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']

            # Take reverse step
            d_t = t_2 - t_1

            # trans_t_2 = self._X_Temp_euler_step(
            #     d_t, t_1, pred_trans_1, trans_t_1)

            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            # ###########################
            # atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            # X = atoms4.detach().cpu()
            # C = chain_idx.detach().cpu()
            # S = model_out['SEQ'].detach().cpu()
            #
            # bf = model_out['pred_bf'] * 20
            # p = Protein.from_XCSB(X, C, S, bf)
            # saved_path = str(t)+'_.pdb'
            # p.to_PDB(saved_path)

            # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        # return [atom37_traj],None

        X = atom37_traj.detach().cpu()
        C = chain_idx.detach().cpu()
        S = model_out['SEQ'].detach().cpu()
        bf = model_out['pred_bf'] * 20

        return X, C, S, bf

    # def hybrid_binder_side_sample(
    #         self,
    #         model,
    #         batch,
    #         sidechain,
    #
    # ):
    #     num_batch = batch['fixed_mask'].shape[0]
    #     chain_idx = batch['chain_idx']
    #     trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
    #     prot_traj = [(trans_0, rotmats_0)]
    #
    #     # Set-up time
    #     ts = torch.linspace(
    #         self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
    #     t_1 = ts[0]
    #
    #     for t_2 in tqdm(ts[1:]):
    #
    #         trans_t_1, rotmats_t_1 = prot_traj[-1]
    #         batch['trans_t'] = trans_t_1
    #         batch['rotmats_t'] = rotmats_t_1
    #         t = torch.ones((num_batch, 1), device=self._device) * t_1
    #         batch['t'] = t
    #         with torch.no_grad():
    #             model_out = model(batch, recycle=1, is_training=True)
    #
    #
    #
    #         # Process model output.
    #         pred_trans_1 = model_out['pred_trans']
    #         pred_rotmats_1 = model_out['pred_rotmats']
    #
    #         # Take reverse step
    #         d_t = t_2 - t_1
    #
    #         trans_t_2 = self._X_Temp_euler_step(
    #             d_t, t_1, pred_trans_1, trans_t_1)
    #
    #         # trans_t_2 = self._trans_euler_step(
    #         #     d_t, t_1, pred_trans_1, trans_t_1)
    #
    #         def _center(_X):
    #             _X = _X - _X.mean(1, keepdim=True)
    #             return _X
    #
    #         trans_t_2 = _center(trans_t_2)
    #         if self._cfg.self_condition:
    #             batch['trans_sc'] = trans_t_2
    #
    #         rotmats_t_2 = self._rots_euler_step(
    #             d_t, t_1, pred_rotmats_1, rotmats_t_1)
    #
    #         prot_traj.append((trans_t_2, rotmats_t_2))
    #         t_1 = t_2
    #
    #
    #
    #     # We only integrated to min_t, so need to make a final step
    #     t_1 = ts[-1]
    #     trans_t_1, rotmats_t_1 = prot_traj[-1]
    #     batch['trans_t'] = trans_t_1
    #     batch['rotmats_t'] = rotmats_t_1
    #     batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
    #
    #     batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
    #     with torch.no_grad():
    #         model_out = model(batch)
    #     pred_trans_1 = model_out['pred_trans']
    #     pred_rotmats_1 = model_out['pred_rotmats']
    #
    #     prot_traj.append((pred_trans_1, pred_rotmats_1))
    #
    #     # Convert trajectories to atom37.
    #     # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
    #     atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()
    #
    #     print('\n now is designing side chain...')
    #     # sidechain
    #     batch['trans_t'] = pred_trans_1
    #     batch['rotmats_t'] = pred_rotmats_1
    #     batch['bbatoms_t'] = atom37_traj
    #
    #     with torch.no_grad():
    #         sidemodel_out = sidechain(batch)
    #     pred_trans_1 = sidemodel_out['pred_trans']
    #     pred_rotmats_1 = sidemodel_out['pred_rotmats']
    #
    #     prot_traj.append((pred_trans_1, pred_rotmats_1))
    #
    #     # Convert trajectories to atom37.
    #     # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
    #     atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()
    #
    #     X = atom37_traj.detach().cpu()
    #     C = chain_idx.detach().cpu()
    #     S = sidemodel_out['SEQ'].detach().cpu()
    #     bf = model_out['pred_bf'] * 20
    #
    #     return X, C, S, bf

    def hybrid_motif_sample(
            self,
            num_batch,
            num_res:list,
            model,

            chain_idx,
            native_X,
            mode='motif',
            fixed_mask=None,
            training=False,

    ):



        res_mask, _, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)

        #center motif
        motif_X = native_X
        motif_batch = {
            'res_mask': res_mask*fixed_mask,
            'chain_idx': chain_idx,
            'bbatoms':motif_X,
            'fixed_mask': fixed_mask
        }


        motif_X = self._motif_center(motif_batch)['bbatoms']

        # p = Protein.from_XCS(native_X, chain_idx, chain_idx, )
        # p.to_PDB('motif_native_'+str('test_')+'.pdb')
        # p = Protein.from_XCS(motif_X, chain_idx, chain_idx, )
        # p.to_PDB('motif_native_'+str('motif_X')+'.pdb')


        rotmats_m,trans_m, _=self.frame_builder.inverse(motif_X, chain_idx)

        # fix motif area
        diffuse_mask=res_mask*(1-fixed_mask)
        trans_0 = _trans_diffuse_mask(trans_0, trans_m, diffuse_mask)
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_m, diffuse_mask)

        motif_t0 = self.frame_builder(rotmats_0.float(), trans_0, chain_idx)
        p = Protein.from_XCS(motif_t0, chain_idx, chain_idx, )
        p.to_PDB('motif_native_'+str('motif_t0')+'.pdb')


        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,
        'fixed_mask': fixed_mask}



        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            # p = Protein.from_XCS(bb_atoms_t, chain_idx, chain_idx, )
            # p.to_PDB('motif_native_test'+str(t_1)+'.pdb')


            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            # #######test motif fixed area
            # bb_atoms_tss = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
            # fixed_area_n = native_X.cpu() * fixed_mask[..., None, None].cpu()
            # fixed_area_p = bb_atoms_tss.cpu() * fixed_mask[..., None, None].cpu()
            #
            # RMSD = torch.sum((fixed_area_n - fixed_area_p) ** 2, dim=-1)
            # RMSD = torch.sqrt(RMSD).mean()
            # print('gnnout: ',RMSD)


            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            # trans_t_2 = self._motif_center()#_center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        #batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        batch['trans_t'] = pred_trans_1
        batch['rotmats_t'] = pred_rotmats_1
        batch['bbatoms_t'] = atom37_traj




        if training:

            return [atom37_traj],_
        else:
            X=atom37_traj
            C=chain_idx+25
            S=torch.ones_like(C)

            return X, C, S
        #
        #
        # fixed_area_n=native_X.cpu()*fixed_mask[...,None,None].cpu()
        # fixed_area_p = X * fixed_mask[..., None, None].cpu()
        # print('fixed area:  ',fixed_area_p.shape)
        # fixed_area_p=self._batch_ot(fixed_area_p[...,1,:],fixed_area_n[...,1,:],fixed_mask.cpu())
        # print('fixed fixed_area_p:  ',fixed_area_p.shape)
        #
        # RMSD=torch.sum((fixed_area_n[...,1,:]-fixed_area_p)**2,dim=-1)
        # RMSD=torch.sqrt(RMSD.mean(), )
        # print('RMSD:  ',RMSD)
        # return [atom37_traj],_



    def hybrid_binder_sample(
            self,
            model,
            batch,
            num_steps=None,

    ):
        num_batch=batch['fixed_mask'].shape[0]
        chain_idx=batch['chain_idx']
        trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        if num_steps is not None:
            self._sample_cfg.num_timesteps = num_steps
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]


        for t_2 in tqdm(ts[1:] ,leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1,is_training=True)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)


            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            # ###########################
            atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            # X = atoms4.detach().cpu()
            # C = chain_idx.detach().cpu()
            # S = model_out['SEQ'].detach().cpu()
            # #
            # bf = model_out['pred_bf'] * 20
            # p = Protein.from_XCSB(X, C, S, bf)
            # # 获取当前使用的全局seed
            # current_seed = get_global_seed()
            # seed_prefix = f"{current_seed}_" if current_seed is not None else ""
            #
            # saved_path = f"{seed_prefix}{t.detach().cpu().item()}_binder_sample.pdb"
            # p.to_PDB(saved_path)

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        # return [atom37_traj],None


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=model_out['SEQ'].detach().cpu()
        bf=model_out['pred_bf']*20

        return X, C, S,bf

    def hybrid_binder_inverse_noise_sample(
            self,
            model,
            batch,
            inverse_noisy_batch,
            num_steps=None,
            num_opt_steps=500,
            opt_lr=1e-3
    ):
        num_batch = batch['fixed_mask'].shape[0]
        chain_idx = batch['chain_idx']



        # --- 阶段1：通过反解噪声获得初始噪声 (NEW)
        optimized_trans_T, optimized_rotmats_T = invert_noise_via_optimization(
            model, inverse_noisy_batch, num_opt_steps=num_opt_steps, lr=opt_lr
        )

        # 使用反解噪声作为采样起点
        trans_0, rotmats_0 = optimized_trans_T, optimized_rotmats_T
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        if num_steps is not None:
            self._sample_cfg.num_timesteps = num_steps
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t

            with torch.no_grad():
                model_out = model(batch, recycle=1, is_training=True)

            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']

            d_t = t_2 - t_1

            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # Final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)

        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        X = atom37_traj.detach().cpu()
        C = chain_idx.detach().cpu()
        S = model_out['SEQ'].detach().cpu()
        bf = model_out['pred_bf'] * 20

        return X, C, S, bf

    def hybrid_binder_side_sample(
            self,
            model,
            batch,
            sidechain,

    ):
        num_batch=batch['fixed_mask'].shape[0]
        chain_idx=batch['chain_idx']
        trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]


        for t_2 in ts[1:]:

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1,is_training=True)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            # trans_t_2 = self._trans_euler_step(
            #     d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            # ###########################
            # atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            # X = atoms4.detach().cpu()
            # C = chain_idx.detach().cpu()
            # S = model_out['SEQ'].detach().cpu()
            #
            # bf = model_out['pred_bf'] * 20
            # p = Protein.from_XCSB(X, C, S, bf)
            # saved_path = str(t)+'_.pdb'
            # p.to_PDB(saved_path)

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        print('\n now is designing side chain...')
        # sidechain
        batch['trans_t'] = pred_trans_1
        batch['rotmats_t'] = pred_rotmats_1
        batch['bbatoms_t'] = atom37_traj


        with torch.no_grad():
            sidemodel_out =    sidechain(batch)
        pred_trans_1 = sidemodel_out['pred_trans']
        pred_rotmats_1 = sidemodel_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=sidemodel_out['SEQ'].detach().cpu()
        bf=model_out['pred_bf']*20

        return X, C, S,bf
    def hybrid_binder_side_sample_inter(
            self,
            model,
            batch,
            sidechain,

    ):
        num_batch=batch['fixed_mask'].shape[0]
        chain_idx=batch['chain_idx']
        trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]


        for t_2 in tqdm(ts[1:]):

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1,is_training=False)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            # trans_t_2 = self._trans_euler_step(
            #     d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            with torch.no_grad():
                batch['trans_t'] = trans_t_2
                batch['rotmats_t'] = rotmats_t_2
                sidemodel_out = sidechain(batch,is_training=True)
            pred_trans_1 = sidemodel_out['pred_trans']
            pred_rotmats_1 = sidemodel_out['pred_rotmats']
            S = sidemodel_out['SEQ'].detach()
            bf = model_out['pred_bf'] * 20
            chi=sidemodel_out['pred_chi']

            batch['chi']=chi
            batch['aatype'] = S
            batch['atoms14_b_factors'] = bf

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch,recycle=1,is_training=False)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=sidemodel_out['SEQ'].detach().cpu()
        bf=model_out['pred_bf']*20

        return X, C, S,bf
    def hybrid_side_sample(
            self,
            model,
            batch

    ):

        chain_idx=batch['chain_idx']

        with torch.no_grad():
            model_out = model(batch,is_training=True)


        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']



        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=model_out['SEQ'].detach().cpu()
        bf=batch['atoms14_b_factors'][...,:4]

        return (X, C, S,bf),model_out



    def hybrid_motif_long_sample(
            self,
            num_batch,
            num_res:list,
            model,
            intervals,
            native_X,
            mode='motif',


    ):


        def make_fixed_mask(mode):

            if mode=='motif':
                # 创建一个全0的tensor
                fixed_maskmask = torch.zeros(num_res, dtype=torch.int)

                # 遍历输入列表，将指定区间和单独位置的值设置为1
                for item in intervals:
                    if isinstance(item, list):  # 如果是区间
                        start, end = item
                        fixed_maskmask[start - 1:end] = 1
                    else:  # 如果是单独的位置
                        fixed_maskmask[item - 1] = 1
            else:
                fixed_maskmask = torch.zeros(num_res, dtype=torch.int)


            return fixed_maskmask.to(self._device)



        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)


        C=chain_idx+25
        S=torch.ones_like(C)


        p = Protein.from_XCS(native_X, C, S, )
        p.to_PDB('1a0aA00_motif_native.pdb')


        # print('make fixed mask by interval')
        fixed_mask = make_fixed_mask(mode)
        init_backbone = self.frame_builder(rotmats_0.float(), trans_0, chain_idx)
        init_backbone = init_backbone * (1 - fixed_mask[..., None, None]) + native_X * fixed_mask[..., None, None]

        p = Protein.from_XCS(init_backbone, C, S, )
        p.to_PDB('1a0aA00_motif_init_backbone.pdb')

        # centrelise
        batch={'res_mask':res_mask,'bbatoms':init_backbone,}
        native_X=self._center(batch)['bbatoms']


        p = Protein.from_XCS(native_X, C, S, )
        p.to_PDB('1a0aA00_motif_centrelise.pdb')


        # get new trans0
        rotmats_f,trans_f, _=self.frame_builder.inverse(native_X, chain_idx)


        # update with fixed mask
        # trans_0 = trans_0*(1-fixed_mask[...,None])+trans_f*fixed_mask[...,None]
        # rotmats_0 = rotmats_0*(1-fixed_mask[...,None,None])+rotmats_f*fixed_mask[...,None,None]

        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,}

        ####### motif part##############
        batch['fixed_mask'] = fixed_mask
        # batch['fixed_mask'] = None
        # batch['trans_1'] = trans_f / 10.0
        # batch['rotmats_1'] = rotmats_f

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_f, rotmats_f)]
        clean_traj = []
        for t_2 in tqdm(ts[1:]):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]


            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            #batch['bbatoms_t'] = bb_atoms_t*(1-fixed_mask[...,None,None])+native_X*fixed_mask[...,None,None]
            batch['bbatoms_t'] =bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)  #

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            # trans_t_2 = self._X_Temp_motif_euler_step(
            #     d_t, t_1,trans_f, pred_trans_1, trans_t_1,fixed_mask,0.1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            # trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            # rotmats_t_2 = self._rots_motif_euler_step(
            #     d_t, t_1,rotmats_f, pred_rotmats_1, rotmats_t_1,fixed_mask,0.1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)



            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2


            #######test motif fixed area
            bb_atoms_tss = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            fixed_area_n = trans_f.cpu() * fixed_mask[..., None].cpu()
            fixed_area_p = trans_t_2.cpu() * fixed_mask[..., None].cpu()

            # p = Protein.from_XCS(bb_atoms_tss, C, S, )
            # p.to_PDB('1a0aA00_motif_'+str(t_2)+'.pdb')

            RMSD = torch.sum((fixed_area_n - fixed_area_p) ** 2, dim=-1)
            RMSD = torch.sqrt(RMSD).mean()
            print('eluer out: ',RMSD)


        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        # pred_trans_1 = _center(pred_trans_1)
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        X=atom37_traj
        C=chain_idx+25
        S=torch.ones_like(C)


        fixed_area_n=trans_f.cpu()*fixed_mask[...,None].cpu()
        fixed_area_p = pred_trans_1.cpu() * fixed_mask[..., None].cpu()

        RMSD=torch.sum((fixed_area_n-fixed_area_p)**2,dim=-1)
        RMSD=torch.sqrt(RMSD.mean(), )
        print(RMSD)
        return X,C,S





def _debug_viz_gradients(
    pml_file, X_list, dX_list, C, S, arrow_length=2.0, name="gradient", color="red"
):
    """ """
    lines = [
        "from pymol.cgo import *",
        "from pymol import cmd",
        f'color_1 = list(pymol.cmd.get_color_tuple("{color}"))',
        'color_2 = list(pymol.cmd.get_color_tuple("blue"))',
    ]

    with open(pml_file, "w") as f:
        for model_ix, X in enumerate(X_list):
            print(model_ix)
            lines = lines + ["obj_1 = []"]

            dX = dX_list[model_ix]
            scale = dX.norm(dim=-1).mean().item()
            X_i = X
            X_j = X + arrow_length * dX / scale

            for a_ix in range(4):
                for i in range(X.size(1)):
                    x_i = X_i[0, i, a_ix, :].tolist()
                    x_j = X_j[0, i, a_ix, :].tolist()
                    lines = lines + [
                        f"obj_1 = obj_1 + [CYLINDER] + {x_i} + {x_j} + [0.15]"
                        " + color_1 + color_1"
                    ]
            lines = lines + [f'cmd.load_cgo(obj_1, "{name}", {model_ix+1})']
            f.write("\n" + "\n".join(lines))
            lines = []


def _debug_viz_XZC(X, Z, C, rgb=True):
    from matplotlib import pyplot as plt

    if len(X.shape) > 3:
        X = X.reshape(X.shape[0], -1, 3)
    if len(Z.shape) > 3:
        Z = Z.reshape(Z.shape[0], -1, 3)
    if C.shape[1] != X.shape[1]:
        C_expand = C.unsqueeze(-1).expand(-1, -1, 4)
        C = C_expand.reshape(C.shape[0], -1)

    # C_mask = expand_chain_map(torch.abs(C))
    # X_expand = torch.einsum('nix,nic->nicx', X, C_mask)
    # plt.plot(X_expand[0,:,:,0].data.numpy())
    N = X.shape[1]
    Ymax = torch.max(X[0, :, 0]).item()
    plt.figure(figsize=[12, 4])
    plt.subplot(2, 1, 1)

    plt.bar(
        np.arange(0, N),
        (C[0, :].data.numpy() < 0) * Ymax,
        width=1.0,
        edgecolor=None,
        color="lightgrey",
    )
    if rgb:
        plt.plot(X[0, :, 0].data.numpy(), "r", linewidth=0.5)
        plt.plot(X[0, :, 1].data.numpy(), "g", linewidth=0.5)
        plt.plot(X[0, :, 2].data.numpy(), "b", linewidth=0.5)
        plt.xlim([0, N])
        plt.grid()
        plt.title("X")
        plt.xticks([])
        plt.subplot(2, 1, 2)
        plt.plot(Z[0, :, 0].data.numpy(), "r", linewidth=0.5)
        plt.plot(Z[0, :, 1].data.numpy(), "g", linewidth=0.5)
        plt.plot(Z[0, :, 2].data.numpy(), "b", linewidth=0.5)
        plt.plot(C[0, :].data.numpy(), "orange")
        plt.xlim([0, N])
        plt.grid()
        plt.title("RInverse @ [X]")
        plt.xticks([])
        plt.savefig("xzc.pdf")
    else:
        plt.plot(X[0, :, 0].data.numpy(), "k", linewidth=0.5)
        plt.xlim([0, N])
        plt.grid()
        plt.title("X")
        plt.xticks([])
        plt.subplot(2, 1, 2)
        plt.plot(Z[0, :, 0].data.numpy(), "k", linewidth=0.5)
        plt.plot(C[0, :].data.numpy(), "orange")
        plt.xlim([0, N])
        plt.grid()
        plt.title("Inverse[X]")
        plt.xticks([])
        plt.savefig("xzc.pdf")
    exit()


