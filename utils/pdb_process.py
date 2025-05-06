
import numpy as np
import os
from omegaconf import DictConfig
import torch
import torch.nn.functional as nn


from rfdiffusion import util
import random
import logging

import glob



def parse_pdb(filename, **kwargs):
    """extract xyz coords for all heavy atoms"""
    with open(filename,"r") as f:
        lines=f.readlines()
    return parse_pdb_lines(lines, **kwargs)


def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res, pdb_idx = [],[]
    for l in lines:
        if l[:4] == "ATOM" and l[12:16].strip() == "CA":
            res.append((l[22:26], l[17:20]))
            # chain letter, res num
            pdb_idx.append((l[21:22].strip(), int(l[22:26].strip())))
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [
        (l[21:22].strip(), int(l[22:26].strip()))
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = (
            l[21:22],
            int(l[22:26]),
            " " + l[12:16].strip().ljust(3),
            l[17:20],
        )
        if (chain,resNo) in pdb_idx:
            idx = pdb_idx.index((chain, resNo))
            # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            for i_atm, tgtatm in enumerate(
                util.aa2long[util.aa2num[aa]][:14]
                ):
                if (
                    tgtatm is not None and tgtatm.strip() == atom.strip()
                    ):  # ignore whitespace
                    xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i, idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]

    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,  # cartesian coordinates, [Lx14]
        "mask": mask,  # mask showing which atoms are present in the PDB file, [Lx14]
        "idx": np.array(
            [i[1] for i in pdb_idx]
        ),  # residue numbers in the PDB file, [L]
        "seq": np.array(seq),  # amino acid sequence, [L]
        "pdb_idx": pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6] == "HETATM" and not (ignore_het_h and l[77] == "H"):
                info_het.append(
                    dict(
                        idx=int(l[7:11]),
                        atom_id=l[12:16],
                        atom_type=l[77],
                        name=l[16:20],
                    )
                )
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out["xyz_het"] = np.array(xyz_het)
        out["info_het"] = info_het

    return out


def process_target(pdb_path, parse_hetatom=False, center=True):
    # Read target pdb and extract features.
    target_struct = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

    # Zero-center positions
    ca_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct["xyz"] - ca_center)
    seq_orig = torch.from_numpy(target_struct["seq"])
    atom_mask = torch.from_numpy(target_struct["mask"])
    seq_len = len(xyz)

    # Make 27 atom representation
    xyz_27 = torch.full((seq_len, 27, 3), np.nan).float()
    xyz_27[:, :14, :] = xyz[:, :14, :]

    mask_27 = torch.full((seq_len, 27), False)
    mask_27[:, :14] = atom_mask
    out = {
        "xyz_27": xyz_27,
        "mask_27": mask_27,
        "seq": seq_orig,
        "pdb_idx": target_struct["pdb_idx"],
    }
    if parse_hetatom:
        out["xyz_het"] = target_struct["xyz_het"]
        out["info_het"] = target_struct["info_het"]
    return out