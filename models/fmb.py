from typing import Any
import torch
import torch.nn.functional as F
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from analysis import metrics
from analysis import utils as au
from models.flow_model import FlowModel, FlowModel_binder, FlowModel_binder_sidechain, FlowModel_seqdesign
from models import utils as mu
from data.interpolant import Interpolant_10, create_binder_mask
from data import utils as du
from data import so3_utils
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim.lr_scheduler import StepLR
from openfold.np import residue_constants
from openfold.utils.loss import find_structural_violations, violation_loss
from analysis.mask import design_masks
from data.motif_sample import MotifSampler
from models.noise_schedule import OTNoiseSchedule
from chroma.data.protein import Protein
import torch.nn as nn
from data.kinematics import get_init_xyz

# RL=ReconstructionLosses().cuda()
mse_loss_fn = nn.MSELoss()
ce = nn.CrossEntropyLoss()


def generate_random_list(total_length):
    while True:
        num_parts = random.randint(2, 8)  # Choose a random number of parts between 2 and 8
        parts = []
        remaining = total_length

        for i in range(num_parts - 1):
            if remaining > 10 * (num_parts - i):
                part = random.randint(10, remaining - 10 * (num_parts - i - 1))
            else:
                break
            parts.append(part)
            remaining -= part

        # Add the last part
        if remaining >= 10:
            parts.append(remaining)
            return parts
        # If the last part isn't valid, loop again


def aa_loss_smoothed(
        logits: torch.tensor,
        gt_aatype: torch.tensor,
        seq_mask: torch.tensor,
        aatempweight: float,
        kind: int,
        **kwargs,
):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(gt_aatype, kind).float()

    # Label smoothing
    S_onehot = S_onehot + aatempweight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * seq_mask, dim=-1) / (torch.sum(seq_mask, dim=-1) + 1)

    ##recovery
    pred = torch.argmax(log_probs, -1)
    pred = pred * seq_mask
    true = (gt_aatype * seq_mask).detach().type(torch.int)

    this_correct = ((pred == true).sum() - (1 - seq_mask.detach()).sum())
    thisnods = torch.sum(seq_mask)
    seq_recovery_rate = 100 * this_correct / thisnods

    return loss, loss_av, seq_recovery_rate


class FlowModule(LightningModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        self._eps = 1e-6

        self.design_task = 'motif'

        if 'corrupt_mode' not in self._exp_cfg:
            print('\n  old version no corrupt mode')
            self.model = FlowModel(cfg.model, mode='base')
            # self._exp_cfg.corrupt_mode='base'
        # Set-up vector field prediction model
        else:
            if self._exp_cfg.corrupt_mode == 'binder':
                self.model = FlowModel_binder(cfg.model)
            elif self._exp_cfg.corrupt_mode == 'motif':
                self.model = FlowModel(cfg.model)

            elif self._exp_cfg.corrupt_mode == 'sidechain':
                # self.pretrained_part  = FlowModel_binder(cfg.model)
                self.model = FlowModel_binder_sidechain(cfg.model)
            elif self._exp_cfg.corrupt_mode == 'base':  # base for mononer or complex
                self.model = FlowModel(cfg.model, mode='base')
            elif self._exp_cfg.corrupt_mode == 'fbb':  # base for mononer or complex
                self.model = FlowModel_seqdesign(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant_10(cfg.interpolant)

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        # Noise schedule
        self.noise_schedule = OTNoiseSchedule()


    def predict_step_motif(self, batch, batch_idx):

        self.interpolant._sample_cfg.num_timesteps = 500
        methods = 'cvode'
        print(self._output_dir)

        ref_pdb = '/home/junyu/project/motif/4jhw.pkl'
        domain = ref_pdb.split('/')[-1].split('.')[0]
        ref_data = du.read_pkl(ref_pdb)
        input_str = '10-25,F196-212,15-30,F63-69, 10-25'

        # 使用当前时间戳设置种子值
        # random.seed(time.time())
        total_length = random.randint(60, 90)

        sampler = MotifSampler(input_str, total_length)
        results = sampler.get_results()
        print(f"Letter segments: {results['letter_segments']}")
        print(f"Number segments: {results['number_segments']}")
        print(f"Total motif length: {results['total_motif_length']}")
        print(f"Random sample total length: {results['random_sample_total_length']}")
        print(f"Sampled lengths: {results['sampled_lengths']}")
        final_output = sampler.get_final_output()
        print(f"Final output: {final_output}")

        designname = ref_pdb.split('/')[-1].split('.')[0]
        self._sample_write_dir = self._output_dir + f'/motifdesign_highnocoil_{designname}_{methods}_temp' + str(
            self.interpolant._cfg.temp) + '_' + str(self.interpolant._sample_cfg.num_timesteps) + '/'

        # self._sample_write_dir='/home/junyu/project/monomer_test/base_neigh/rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto/2024-03-12_12-52-05/last_256/rcsb/motif_1bcf/'
        os.makedirs(self._sample_write_dir, exist_ok=True)
        os.makedirs(self._sample_write_dir + '/native/', exist_ok=True)
        os.makedirs(self._sample_write_dir + '/motif_masks/', exist_ok=True)

        from .Proflow import process_input
        from chroma.data.protein import Protein
        import tqdm

        device = f'cuda:{torch.cuda.current_device()}'

        interpolant = self.interpolant
        interpolant.set_device(device)

        # native structure
        positions = np.take(ref_data['atom_positions'], ref_data['modeled_idx'], axis=0)[..., [0, 1, 2, 4], :]
        chain_index = np.take(ref_data['chain_index'], ref_data['modeled_idx'], axis=0)
        aatype = np.take(ref_data['aatype'], ref_data['modeled_idx'], axis=0)
        p = Protein.from_XCS(torch.tensor(positions).unsqueeze(0), torch.tensor(chain_index).unsqueeze(0),
                             torch.tensor(aatype).unsqueeze(0))
        p.to_PDB(self._sample_write_dir + f'/native/{domain}_motif_native.pdb')

        _, _, _, indices_mask = process_input(final_output, ref_data)
        np.savetxt(self._sample_write_dir + f'/motif_masks/motif_native.npy', indices_mask)

        for i in tqdm.tqdm(range(100)):
            total_length = random.randint(60, 90)
            sampler = MotifSampler(input_str, total_length)
            results = sampler.get_results()
            print(f"Letter segments: {results['letter_segments']}")
            print(f"Number segments: {results['number_segments']}")
            print(f"Total motif length: {results['total_motif_length']}")
            print(f"Random sample total length: {results['random_sample_total_length']}")
            print(f"Sampled lengths: {results['sampled_lengths']}")
            final_output = sampler.get_final_output()
            print(f"Final output: {final_output}")

            init_motif, fixed_mask, aa_motifed, indices_mask = process_input(final_output, ref_data)

            chain_idx = torch.ones_like(fixed_mask).unsqueeze(0)
            fixed_mask = fixed_mask.unsqueeze(0)
            bbatoms = torch.tensor(init_motif)[..., [0, 1, 2, 4], :].unsqueeze(0).to(self.device).float()
            sample_length = bbatoms.shape[1]

            X, C, S = self.interpolant.hybrid_motif_sample(
                1,
                [sample_length],
                self.model,
                chain_idx=chain_idx.to(self.device),
                native_X=bbatoms.to(self.device),
                mode='motif',
                fixed_mask=fixed_mask.to(self.device),

            )

            native_aatype = torch.tensor(aa_motifed).unsqueeze(0).to(S.device)
            S = native_aatype * fixed_mask.to(S.device)
            p = Protein.from_XCS(X, C, S, )

            # 输出到文本文件
            p.to_PDB(self._sample_write_dir + f'/{domain}_motif' + str(i) + '.pdb')
            np.savetxt(self._sample_write_dir + f'/motif_masks/{domain}_motif' + str(i) + '.npy',
                       fixed_mask.cpu().numpy())

            # positions = torch.nonzero(fixed_mask.squeeze(0) == 1).squeeze(-1)
            # # 转换为列表
            # positions_list = positions.tolist()
            # # 转换为逗号分隔的字符串
            # positions_str = ', '.join(map(str, positions_list))
            # with open(self._sample_write_dir + '/5yui_motif_info.txt', 'w') as f:
            #
            #         f.write(f"> 5yui_motif, fixed area \n")
            #         f.write(f"> in native \n")
            #         f.write(f"> {parama}  \n")
            #         f.write(f"> in design \n")
            #         f.write(f"{positions_str}\n")

