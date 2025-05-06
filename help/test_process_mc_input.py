import re
import numpy as np
import torch
from data import utils as du
from data.m2 import MotifSampler,MotifSamplerMultiChain
from data.protein import PDB_CHAIN_IDS,chain_ids_to_sequence
from data.utils import ALPHANUMERIC
import random

def process_input(input_str, pdb_dict):
    print(input_str)
    chain_segments = input_str.split('/')
    all_positions = []
    mask = []
    aatype = []
    indices_mask = np.zeros_like(pdb_dict['modeled_idx'])
    chain_ids = []

    for segment in chain_segments:
        chain_intervals = re.split(r',\s*', segment)
        current_chain = None

        # 确定当前链字符
        current_chain = None
        for interval in chain_intervals:
            match = re.match(r'([A-Za-z])(\d+-\d+)', interval)
            if match:
                current_chain = match.group(1)
                break

        if current_chain is None:
            raise ValueError(f"Chain character missing for segment: {segment}")

        for interval in chain_intervals:
            if re.match(r'^[A-Za-z]\d+-\d+$', interval):  # 处理包含链字符的区间
                chain_char, start, end = re.findall(r'([A-Za-z])(\d+)-(\d+)', interval)[0]
                start, end = int(start), int(end)
                chain_int = du.chain_str_to_int(chain_char)

                indices = np.where((pdb_dict['chain_index'] == chain_int) &
                                   (pdb_dict['residue_index'] >= start) &
                                   (pdb_dict['residue_index'] <= end))[0]
                positions = np.take(pdb_dict['atom_positions'], indices, axis=0)
                all_positions.append(positions)
                indices_mask[indices] = 1
                seqs_motif = np.take(pdb_dict['aatype'], indices, axis=0)
                aatype.append(seqs_motif)

                length = end - start + 1
                mask.extend([1] * length)
                chain_ids.extend([ALPHANUMERIC[chain_int]] * length)
            elif re.match(r'^\d+-\d+$', interval):  # 区间范围
                start, end = map(int, interval.split('-'))
                length = random.randint(start, end)
                random_positions = np.random.rand(length, 37, 3)
                all_positions.append(random_positions)
                aatype.append(np.zeros(length))
                mask.extend([0] * length)
                chain_ids.extend([current_chain] * length)
            elif re.match(r'^\d+$', interval):  # 固定长度
                length = int(interval)
                random_positions = np.random.rand(length, 37, 3)
                all_positions.append(random_positions)
                aatype.append(np.zeros(length))
                mask.extend([0] * length)
                chain_ids.extend([current_chain] * length)
            else:
                raise ValueError(f"Unrecognized interval format: {interval}")

    mask_tensor = torch.tensor(mask, dtype=torch.int)
    chain_ids=chain_ids_to_sequence(chain_ids,PDB_CHAIN_IDS)
    return np.concatenate(all_positions, axis=0), mask_tensor, np.concatenate(aatype, axis=0), indices_mask, chain_ids


if __name__ == '__main__':

    # 示例使用
    input_str = "E400-510/20-45,A24-42,4-10,A64-82,0-5"
    ref_data = du.read_pkl('/home/junyu/project/motif/rf_pdb_pkl/6vw1.pkl')
    result = process_input(MotifSamplerMultiChain(input_str).get_final_output(), ref_data)
    print(result)
