import torch
import numpy as np


# ideal N, CA, C initial coordinates
init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
INIT_CRDS = torch.full((4, 3), np.nan)
INIT_CRDS[:3] = torch.stack((init_N, init_CA, init_C), dim=0) # (3,3)

def get_init_xyz(xyz_t):
    # input: xyz_t (B, T, L, 14, 3)
    # ouput: xyz (B, T, L, 14, 3)
    B, T, L = xyz_t.shape[:3]
    init = INIT_CRDS.to(xyz_t.device).reshape(1,1,1,4,3).repeat(B,T,L,1,1)
    if torch.isnan(xyz_t).all():
        return init

    mask = torch.isnan(xyz_t[:,:,:,:3]).any(dim=-1).any(dim=-1) # (B, T, L)
    #
    center_CA = ((~mask[:,:,:,None]) * torch.nan_to_num(xyz_t[:,:,:,1,:])).sum(dim=2) / ((~mask[:,:,:,None]).sum(dim=2)+1e-4) # (B, T, 3)
    xyz_t = xyz_t - center_CA.view(B,T,1,1,3)
    #
    idx_s = list()
    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            exist_in_templ = torch.where(~mask[i_b, i_T])[0] # (L_sub)
            seqmap = (torch.arange(L, device=xyz_t.device)[:,None] - exist_in_templ[None,:]).abs() # (L, L_sub)
            seqmap = torch.argmin(seqmap, dim=-1) # (L)
            idx = torch.gather(exist_in_templ, -1, seqmap) # (L)
            offset_CA = torch.gather(xyz_t[i_b, i_T, :, 1, :], 0, idx.reshape(L,1).expand(-1,3))
            init[i_b,i_T] += offset_CA.reshape(L,1,3)
    #
    xyz = torch.where(mask.view(B, T, L, 1, 1), init, xyz_t)
    return xyz