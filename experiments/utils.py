"""Utility functions for experiments."""
import logging
import torch
import os
import numpy as np
from analysis import utils as au
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import Dataset

# 添加一个全局变量来存储seed
GLOBAL_SEED = None

def set_global_seed(seed=42):
    """设置全局随机种子并将其存储在环境变量中"""
    # 将种子存储在环境变量中
    os.environ['GLOBAL_RANDOM_SEED'] = str(seed)
    
    import random
    import numpy as np
    import torch

    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)

    # Set CUDA's random seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Additional settings for CUDA determinism
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global random seed set to {GLOBAL_SEED}")

def get_global_seed():
    """从环境变量中获取全局随机种子"""
    seed_str = os.environ.get('GLOBAL_RANDOM_SEED')
    return int(seed_str) if seed_str is not None else None

class CSVDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.iloc[idx, :-1].values.astype(float)
        label = self.dataframe.iloc[idx, -1]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [
                int(x) for x in samples_cfg.length_subset
            ]
        all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(self._samples_cfg.samples_per_length):
                all_sample_ids.append((length, sample_id))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': num_res,
            'sample_id': sample_id,
        }
        return batch



def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aatype = None,
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_path = au.save_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aatype,
    )
    # prot_traj_path = au.write_prot_to_pdb(
    #     bb_prot_traj,
    #     prot_traj_path,
    #     b_factors=b_factors,
    #     no_indexing=True,
    #     aatype=aatype,
    # )
    # x0_traj_path = au.write_prot_to_pdb(
    #     x0_traj,
    #     x0_traj_path,
    #     b_factors=b_factors,
    #     no_indexing=True,
    #     aatype=aatype
    # )
    return {
        'sample_path': sample_path,
        # 'traj_path': prot_traj_path,
        # 'x0_traj_path': x0_traj_path,
    }


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened
