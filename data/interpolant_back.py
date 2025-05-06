import torch

import numpy as np
from tqdm.auto import tqdm

from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy

from scipy.optimize import linear_sum_assignment



from chroma.layers.structure.mvn import BackboneMVNGlobular,BackboneMVNResidueGas

from chroma.layers.structure.backbone import FrameBuilder
from models.noise_schedule import OTNoiseSchedule


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

    def _corrupt_trans(self, trans_1, t,rg, res_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, None, self._device)
        trans_0 = trans_nm_0 * rg
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _corrupt_trans_rg(self, trans_1, t, res_mask):

        batch,nres = res_mask.shape
        trans_0=generate_batch_constrained_points_torch(batch,nres,device=self._device)

        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
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

    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
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
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def _center(self,batch):

        # old=batch['bbatoms'][0]
        # name=batch['csv_idx'][0].detach().cpu().numpy()
        # save_pdb(old.detach().cpu().numpy().reshape(-1,3),str(name)+'_old.pdb')

        bb_pos = batch['bbatoms'][:,:, 1]
        bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'],dim=1) + 1e-5)[:,None]
        #aa = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'], dim=1) + 1e-5)[ None]
        batch['bbatoms'] = batch['bbatoms'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0].detach().cpu().numpy().reshape(-1, 3), str(name)+'_new.pdb')
        return batch
    def corrupt_batch(self, batch):
        #batch=self._center(batch)
        noisy_batch = copy.deepcopy(batch)

        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        bbatoms = batch['bbatoms']#.float() # Angstrom
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
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        return noisy_batch
    def corrupt_batch_rg(self, batch):
        noisy_batch = copy.deepcopy(batch)

        bbatoms = batch['bbatoms']  # Angstrom
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
        rg = (2 / 3) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans_rg(trans_1, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        return noisy_batch

    def corrupt_batch_atoms(self, batch):
        '''
        corrupt bbatoms
        0 is the ground truth
        1 is the noisy

        '''
        noisy_batch = copy.deepcopy(batch)

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        # [B, N]
        chain_idx = batch['chain_idx']

        # [B, N, 3]
        bbatoms = batch['bbatoms']  # Angstrom
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system

        noisy_batch['rotmats_1'] = gt_rotmats_1
        noisy_batch['trans_1'] = gt_trans_1

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        bbatoms_t = self._corrupt_bbatoms(bbatoms, chain_idx, t, res_mask)
        noisy_batch['bbatoms_t'] = bbatoms_t

        rotmats_t, trans_t, _q = self.frame_builder.inverse(bbatoms_t, chain_idx)  # frames in new rigid system

        noisy_batch['rotmats_t'] = rotmats_t.float()
        noisy_batch['trans_t'] = trans_t.float()

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
        temp = self._cfg.temp
        h=1/t
        g2=(2-2*t)/t
        score=(t*trans_1-trans_t)/(1-t)**2
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

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss) * (
                t[:, 0] > training_cfg.aux_loss_t_pass
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
            'chain_idx': chain_idx,}

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

        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return [atom37_traj], clean_atom37_traj, clean_traj



    def hybrid_sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,}

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
            trans_t_2 = self._X_Temp_euler_step(
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

        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return [atom37_traj], clean_atom37_traj, clean_traj



    def cp_sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = generate_batch_constrained_points_torch(num_batch,
            num_res,device=self._device)

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,}

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        entropy=[]
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
                model_out,e = model(batch)
            entropy.append(e)
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
            model_out,e = model(batch)

        entropy.append(e)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return [atom37_traj], clean_atom37_traj, clean_traj,entropy

    def dw(self,num_batch,num_res,chain_idx):
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device)

        z = torch.rand(num_batch, num_res, 4, 3).to(trans_0.device)
        mask = torch.zeros(num_batch, num_res, 4, ).to(trans_0.device)
        mask[..., 1] = 1
        others_z = z * (1 - mask[..., None])  ##nm _SCALE
        z = mask[..., None] * trans_0.unsqueeze(-2).repeat(1, 1, 4, 1) + others_z
        bb_atoms_t = self.backbone_init.sample(chain_idx, Z=z)

        return bb_atoms_t
    def X_sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device)

        z = torch.rand(num_batch, num_res, 4, 3).to(trans_0.device)
        mask = torch.zeros(num_batch, num_res, 4, ).to(trans_0.device)
        mask[..., 1] = 1
        others_z = z * (1 - mask[..., None])  ##nm _SCALE

        # others_1 = trans_1 - trans_1[..., 1, :].unsqueeze(-2).repeat(1, 1, 4, 1)  #ANG_SCALE
        # others_1 = others_1/ du.NM_TO_ANG_SCALE  # nm_SCALE
        # others_t = (1 - t[..., None]) * others_z + t[..., None] * others_1  # nm_SCALE

        # 3. combine and transform to resgas
        self.backbone_init._register(stddev_CA=rg, device=self._device)
        z = mask[..., None] * trans_0.unsqueeze(-2).repeat(1, 1, 4, 1) + others_z
        bb_atoms_t = self.backbone_init.sample(chain_idx, Z=z)

        rotmats_0, trans_0, _q = self.frame_builder.inverse(bb_atoms_t,
                                                                chain_idx)  # frames in new rigid system


        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,}

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

            bb_atoms_t_pred = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx)


            # Take reverse step
            d_t = t_2 - t_1
            bb_atoms_t_pred2 = self._X_Temp_euler_step(
                d_t, t_1, bb_atoms_t_pred, bb_atoms_t)

            # bb_atoms_t_pred2 = self.sde(
            #     d_t, t_1, bb_atoms_t_pred, bb_atoms_t,chain_idx)

            def _center(_X):
                ca = _X[..., :, 1, :]
                _X = _X - ca.mean(1, keepdim=True)[..., :, None, :]
                return _X

            bb_atoms_t_pred2 = _center(bb_atoms_t_pred2)


            rotmats_t_2, trans_t_2, _q = self.frame_builder.inverse(bb_atoms_t_pred2, chain_idx)  # frames in new rigid system
            bb_atoms_t=bb_atoms_t_pred2.float()


            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2.float()


            prot_traj.append((trans_t_2.float(), rotmats_t_2.float()))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = bb_atoms_t
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

        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return [atom37_traj], clean_atom37_traj, clean_traj

    def H_sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)



        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,}

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]
        bb_atoms_t = self.frame_builder(rotmats_0, trans_0, chain_idx)
        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        i=0
        for t_2 in ts[1:]:
            d_t=t_2-t_1
            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if i%2==0:
                trans_t_2, rotmats_t_2,bb_atoms_t = self.update_SE3_space(model, num_batch, batch, t_1, d_t, trans_t_1,
                                                               rotmats_t_1, bb_atoms_t, chain_idx)

            else:
                trans_t_2, rotmats_t_2 , bb_atoms_t=self.update_X_space(
                    model, num_batch, batch, t_1, d_t, trans_t_1, rotmats_t_1, bb_atoms_t,chain_idx)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2
            i=i+1

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

        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return [atom37_traj], clean_atom37_traj, clean_traj

    def update_SE3_space(self,model,num_batch,batch,t_1,d_t,trans_t_1,rotmats_t_1,bb_atoms_t,chain_idx):
        # Run model.

        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = bb_atoms_t
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        if self._cfg.self_condition:
            batch['trans_sc'] = trans_t_1

        with torch.no_grad():
            model_out = model(batch)

        # Process model output.
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']


        # Take reverse step

        trans_t_2 = self._X_Temp_euler_step(
            d_t, t_1, pred_trans_1, trans_t_1)

        def _center(_X):
            _X = _X - _X.mean(1, keepdim=True)
            return _X

        trans_t_2 = _center(trans_t_2)


        rotmats_t_2 = self._rots_euler_step(
            d_t, t_1, pred_rotmats_1, rotmats_t_1)

        bb_atoms_t_pred = self.frame_builder(rotmats_t_2, trans_t_2, chain_idx)

        return trans_t_2, rotmats_t_2,bb_atoms_t_pred

    def update_X_space(self,model,num_batch,batch,t_1,d_t,trans_t_1,rotmats_t_1,bb_atoms_t,chain_idx):
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1


        batch['bbatoms_t'] = bb_atoms_t

        t = torch.ones((num_batch, 1), device=self._device) * t_1
        batch['t'] = t
        if self._cfg.self_condition:
            batch['trans_sc'] = trans_t_1
        with torch.no_grad():
            model_out = model(batch)

        # Process model output.
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']


        bb_atoms_t_pred = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx)

        # Take reverse step

        bb_atoms_t_pred2 = self._X_Temp_euler_step(
            d_t, t_1, bb_atoms_t_pred, bb_atoms_t)

        def _center(_X):
            ca = _X[..., :, 1, :]
            _X = _X - ca.mean(1, keepdim=True)[..., :, None, :]
            return _X

        bb_atoms_t_pred2 = _center(bb_atoms_t_pred2)

        rotmats_t_2, trans_t_2, _q = self.frame_builder.inverse(bb_atoms_t_pred2,
                                                                chain_idx)  # frames in new rigid system
        bb_atoms_t = bb_atoms_t_pred2.float()



        return trans_t_2, rotmats_t_2,bb_atoms_t


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


