"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
import collections
from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from torch.utils.data import SequentialSampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import pickle
# 你可以根据你的配置文件来初始化 PdbDataModule
import random
import math
from collections import defaultdict


import data.residue_constants as rc
from chroma.layers.structure.sidechain import ChiAngles,SideChainBuilder
atom14tochis=ChiAngles()
chitoatoms=SideChainBuilder()




class LengthSpecificDataset(Dataset):
    def __init__(self, original_dataset, selected_lengths):
        self.original_dataset = original_dataset
        self.filtered_indices = self._filter_by_length(selected_lengths)

    def _filter_by_length(self, selected_lengths):
        filtered_indices = []
        # 创建一个字典来存储每个长度的样本索引
        length_indices = {length: [] for length in selected_lengths}

        # 遍历原始数据集，根据长度分类样本索引
        for idx, sample in enumerate(self.original_dataset):
            sample_length = len(sample['aatype'])
            if sample_length in selected_lengths:
                length_indices[sample_length].append(idx)

        # 每个长度随机选择一个样本
        for length in selected_lengths:
            if length_indices[length]:
                selected_idx = random.choice(length_indices[length])
                filtered_indices.append(selected_idx)

        return filtered_indices

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        return self.original_dataset[original_idx]
class sLengthBatcher:
    def __init__(self, dataset, max_batch_size, max_num_res_squared, seed=123, shuffle=True):
        self._log = logging.getLogger(__name__)
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_num_res_squared = max_num_res_squared
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.sample_order = []
        self._create_batches()

    def _create_batches(self):
        # 根据序列长度对数据集进行分组
        length_groups = defaultdict(list)
        for idx, item in enumerate(self.dataset):
            seq_len = len(item['aatype'])
            length_groups[seq_len].append(idx)
        max_values_per_seq_len={}
        # 创建批次

        sample_order = []
        for seq_len, indices in length_groups.items():
            max_batch_size = min(self.max_batch_size, self.max_num_res_squared // (seq_len ** 2) + 1)
            max_values_per_seq_len[seq_len]=max_batch_size*seq_len



            num_batches = math.ceil(len(indices) / max_batch_size)
            for i in range(num_batches):
                batch_indices = indices[i:i + max_batch_size]
                sample_order.append(batch_indices)


        # 随机排列批次以消除长度偏见
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            self.sample_order = [sample_order[i] for i in new_order]
        else:
            self.sample_order = sample_order

    def __iter__(self):
        self._create_batches()  # 重新创建批次以每个 epoch 都有不同的顺序
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)

class sPdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.sampler_cfg = data_cfg.sampler

    def setup(self, stage: str = None):
        # 注意：这里假设您已经将数据集整合到了一个 .pkl 文件中
        if stage == 'fit' or stage is None:

            self._log.info(
                f'now is loading training  examples')
            self._train_dataset = sPdbDataset(
                pkl_file_path=self.dataset_cfg.pkl_file_path,  # 修改为正确的.pkl文件路径
                is_training=True,
            )

            self._log.info(
                f'now is loading val  examples')

            # selected_lengths: 从60到512均匀选择10个长度
            selected_lengths = np.linspace(
                self.dataset_cfg.min_num_res,
                self.dataset_cfg.min_eval_length,
                8,  #self.sampler_cfg.max_batch_size
                dtype=int )

            # 创建新的数据集实例
            self._valid_dataset = LengthSpecificDataset(self._train_dataset, selected_lengths)

            self._log.info(
                f'now  loadied {len(self._valid_dataset)} val  examples')


    def train_dataloader(self):
            num_workers = self.loader_cfg.num_workers
            # 使用 LengthBatcher 作为 batch_sampler
            print(len(self._train_dataset))
            batch_sampler = sLengthBatcher(
                dataset=self._train_dataset,
                max_batch_size=self.sampler_cfg.max_batch_size,
                max_num_res_squared=self.sampler_cfg.max_num_res_squared,
                seed=self.dataset_cfg.seed,
                shuffle=True
            )



            return DataLoader(
                self._train_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=False,
                persistent_workers=num_workers > 0
            )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            sampler=SequentialSampler(self._valid_dataset,),
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=True,
        )




class sPdbDataset(Dataset):
    def __init__(self, pkl_file_path, is_training):
        self._log=logging.getLogger(__name__)
        self.is_training = is_training
        self.data = self._load_data(pkl_file_path)

    def _load_data(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # 如果data是defaultdict，说明只有一个数据，将其包装成列表
        if isinstance(data, collections.defaultdict):
            data = [data]
        
        self._log.info(f'read datas of length: {len(data)}')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        item['csv_idx'] = torch.tensor(idx, dtype=torch.long)
        return item



# 创建一个映射字典来将字符转换为数字
mapping_dict = {'NA': 0, 'C': 1, 'E': 2, 'H': 3}
# 使用 numpy 的 vectorize 方法来应用这个映射
vectorized_mapping = np.vectorize(mapping_dict.get)






def generate_atoms14(aatype):
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=aatype.device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=aatype.device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=aatype.device,
    )
    protein_aatype = aatype.to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    return residx_atom14_to_atom37.long(),restype_atom14_mask[protein_aatype]





def gather_atoms(data,inds):
    # 使用 torch.gather 来提取数据
    # 需要确保 inds 的每个值都是要提取的维度的有效索引
    # inds 需要扩展为和 data 相同的最终维度，以便广播
    expanded_inds = inds.unsqueeze(-1).expand(-1, -1, data.size(-1))

    # 提取特征，dim=1 表示沿着第二个维度进行索引
    result = torch.gather(data, 1, expanded_inds)
    return  result

def calculate_interface_residues_v3(all_atom_positions, com_idx, chain_idx, interface_distance_threshold=10.0,cut_length=384):
    """
    使用矩阵操作计算界面残基。
    :param all_atom_positions: 所有原子的位置，尺寸为 [N, 14, 3]。
    :param com_idx: 每个残基所属的蛋白质side标记，尺寸为 [N]。
    :param chain_idx: 每个残基所属的链的标记，尺寸为 [N]。
    :param interface_distance_threshold: 界面残基的距离定义范围。
    :return: 界面上残基的索引列表。
    """
    N = all_atom_positions.size(0)
    CA_positions = all_atom_positions[:, 1, :]  # 假设CA原子的索引为1
    dX = CA_positions.unsqueeze(1) - CA_positions.unsqueeze(0)
    distance_matrix = torch.sqrt(torch.sum(dX ** 2, dim=2))

    # 计算com_idx的相对矩阵，结果不为0的位置表示两个残基不在同一个蛋白质中
    com_idx_matrix = (com_idx[..., None] - com_idx[..., None, :]).abs()
    com_idx_different = com_idx_matrix != 0

    # 找到属于不同蛋白质侧的残基对中距离最小的那对
    min_distance, min_pos = torch.min(distance_matrix[com_idx_different], 0)
    min_pos = min_pos.item()  # 转换为python int

    # 获取这对残基对应的chain_idx
    idx_flat = torch.nonzero(com_idx_different.view(-1))[min_pos]
    chain_a, chain_b = chain_idx[idx_flat // N], chain_idx[idx_flat % N]

    # 确定这两个chain之间距离小于阈值的所有残基对
    chain_a_mask = (chain_idx == chain_a)
    chain_b_mask = (chain_idx == chain_b)

    if chain_a_mask.sum() + chain_b_mask.sum() > cut_length:
        interface_sure = False

        while interface_sure is False:
            interface_mask = (distance_matrix < interface_distance_threshold) & chain_a_mask[:, None] & chain_b_mask[None, :]

            # 提取界面残基的索引
            interface_residues_a = torch.where(interface_mask.any(dim=1))[0].tolist()
            interface_residues_b = torch.where(interface_mask.any(dim=0))[0].tolist()

            # 合并两个链的界面残基索引，去重
            interface_residues = list(set(interface_residues_a + interface_residues_b))

            if len(interface_residues) <= cut_length :
                interface_sure = True

            else:
                interface_distance_threshold = interface_distance_threshold - 1

    else:
        ### use 2 chians
        # 在这里，我们不再需要检查距离矩阵
        interface_mask_a = chain_a_mask
        interface_mask_b = chain_b_mask

        # 提取界面残基的索引
        interface_residues_a = torch.where(interface_mask_a)[0].tolist()
        interface_residues_b = torch.where(interface_mask_b)[0].tolist()


        # 合并两个链的界面残基索引，去重
        interface_residues = list(set(interface_residues_a + interface_residues_b))

    return interface_residues, None


def calculate_interface_residues_vtarget(all_atom_positions, com_idx, fixed_com_idx, n=1):
    """
    使用矩阵操作计算界面残基。
    :param all_atom_positions: 所有原子的位置，尺寸为 [N, 14, 3]。
    :param com_idx: 每个残基所属的蛋白质标记，尺寸为 [N]。
    :param chain_idx: 每个残基所属的链的标记，尺寸为 [N]。
    :param fixed_com_idx: 指定的目标蛋白质标记。
    :param n: 找到与指定 com_idx 最近的 n 个残基。
    :return: 界面上残基的索引列表。
    """
    all_atom_positions=all_atom_positions.cpu()
    com_idx=com_idx.cpu()

    N = all_atom_positions.size(0)
    CA_positions = all_atom_positions[:, 1, :]  # 假设CA原子的索引为1
    dX = CA_positions.unsqueeze(1) - CA_positions.unsqueeze(0)
    distance_matrix = torch.sqrt(torch.sum(dX ** 2, dim=2))

    # 获取目标蛋白质的残基掩码
    #target_mask = (com_idx == fixed_com_idx)

    fixed_com_idx = torch.tensor(fixed_com_idx, device=com_idx.device)
    target_mask = torch.isin(com_idx, fixed_com_idx)

    # 获取非目标蛋白质的残基掩码
    #non_target_mask = (com_idx != fixed_com_idx)
    non_target_mask = ~torch.isin(com_idx, fixed_com_idx)


    # 计算目标蛋白质与非目标蛋白质残基之间的距离矩阵
    target_distances = distance_matrix[target_mask][:, non_target_mask]

    # 获取目标和非目标蛋白质的残基索引
    target_indices = torch.where(target_mask)[0]
    non_target_indices = torch.where(non_target_mask)[0]

    # 获取所有残基对及其距离
    non_target_flat_indices = torch.arange(target_distances.size(1)).repeat(target_distances.size(0))
    all_distances = target_distances.view(-1)

    # 找到距离最近的残基对，直到收集到 n 个不同的非目标蛋白质残基索引
    _, sorted_indices = torch.sort(all_distances)
    closest_non_target_residues = []
    for idx in sorted_indices:
        non_target_residue = non_target_indices[non_target_flat_indices[idx]].item()
        if non_target_residue not in closest_non_target_residues:
            closest_non_target_residues.append(non_target_residue)
        if len(closest_non_target_residues) >= n:
            break

    # from data.interpolant import save_pdb_chain
    # atoms=all_atom_positions[closest_non_target_residues][..., :4, :]
    # chian=com_idx[closest_non_target_residues]

    # save_pdb_chain(atoms.reshape(-1, 3).cpu().numpy(), chian.cpu().numpy(),
    #                f'test_close.pdb')

    return closest_non_target_residues, None


class PdbDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self._init_metadata()
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        self._cut_mode=self._dataset_cfg.cut_mode
        self.cut_length=dataset_cfg.cut_length

        self.min_num_res=self._dataset_cfg.min_num_res

        if self.dataset_cfg.category==None:
            categoryname='all'
        else:
            categoryname=self.dataset_cfg.category

        csv_path=self.dataset_cfg.csv_path
        filepath=csv_path.split('pkl_complex')[0].split('metadata.csv')[0]
        self.pklname=filepath+'reference'
        print(f'will save in {self.pklname}')

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._dataset_cfg.cluster_path, "r") as f:
            for i,line in enumerate(f):
                for chain in line.split(' '):
                    pdb = chain.split('_')[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def _init_metadata(self):
        """Initialize metadata."""


        self._missing_pdbs = 0

        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)


        self.csv = pdb_csv
        self._log.info(
            f'Design: {len(self.csv)} examples with lengths {len(pdb_csv)}')

    def _process_csv_row_complex(self, processed_file_path,idx):
        # 读取并解析处理后的特征
        processed_feats = du.read_pkl(processed_file_path)
        name = processed_file_path.split('/')[-1].split('.')[0]
        # print(name)
        #processed_feats = du.read_pkl('/media/junyu/DATA/mmcif/compedidx/ep/6epg.pkl')


        processed_feats = du.parse_chain_feats(processed_feats)

        # 使用布尔索引筛选出有用的原子位置和链索引
        bool_indices = processed_feats['bb_mask'].astype(bool)
        selected_A = processed_feats['atom_positions'][bool_indices][..., [0, 1, 2, 4], :]
        chain_idx = processed_feats['chain_index'][bool_indices]
        modeled_idx = processed_feats['modeled_idx']
        # 删除不再使用的键
        del processed_feats['modeled_idx']


        # print(bool_indices.sum())
        # for i in processed_feats:
        #     print(i)
        #     xx=processed_feats[i][modeled_idx.tolist()]
        #
        # if processed_feats['ss'].shape[0]!=processed_feats['aatype'].shape[0]:
        #     print(name, ' is not right SS ')


        try:

            if np.max(modeled_idx) >= processed_feats['ss'].shape[0]:
                # 给定的数值，我们想删除所有大于这个数值的元素
                threshold = processed_feats['ss'].shape[0]
                # 创建一个新数组，仅包含不大于threshold的元素
                modeled_idx_ss = modeled_idx[modeled_idx <= threshold]

                processed_featsx = tree.map_structure(lambda x: x[modeled_idx_ss.tolist()], processed_feats)
            else:
                processed_featsx = tree.map_structure(lambda x: x[modeled_idx.tolist()], processed_feats)
            processed_feats=processed_featsx
        except:
            try:
                processed_featsy = tree.map_structure(lambda x: x[bool_indices], processed_feats)
                processed_feats = processed_featsy
            except:
                print('map error: ',name)

        # 转换特征为张量并应用映射
        # chain_feats = {
        #     'aatype': torch.tensor(processed_feats['aatype']).long(),
        #     'com_idx': torch.tensor(processed_feats['com_idx']),
        #     'ss': torch.tensor(vectorized_mapping(processed_feats['ss']))
        # }
        aatype=torch.tensor(processed_feats['aatype']).long()
        com_idx= torch.tensor(processed_feats['com_idx'])

        if torch.all(com_idx == com_idx[0]):

            print(name,' is not p-p complex')
            return []

        ss=torch.tensor(vectorized_mapping(processed_feats['ss']))

        # 获取残基索引和链索引
        res_idx = processed_feats['residue_index']
        chain_idx = processed_feats['chain_index']

        atoms37 = torch.tensor(processed_feats['atom_positions']).float()

        # 获取atoms14相关特征
        residx_atom14_to_atom37, restype_atom14_mask = generate_atoms14(aatype)
        atoms14 = restype_atom14_mask[..., None] * gather_atoms(atoms37, residx_atom14_to_atom37)
        chi, mask_chi = atom14tochis(atoms14.unsqueeze(0), torch.tensor(chain_idx).unsqueeze(0),
                                     aatype.unsqueeze(0))
        chi = chi.squeeze(0)
        mask_chi = mask_chi.squeeze(0)

        atoms14_b_factors = gather_atoms(torch.tensor(processed_feats['b_factors']).unsqueeze(-1),
                                         residx_atom14_to_atom37).squeeze(-1)
        atoms14_mask = gather_atoms(torch.tensor(processed_feats['atom_mask']).unsqueeze(-1),
                                    residx_atom14_to_atom37).squeeze(-1)

        # 根据模型索引处理界面残基
        # in design, we collect all aa, so we need to filter, interface_residues=modeled_idx

        if self._is_training:
            interface_residues, _ = calculate_interface_residues_v3(atoms14, com_idx, torch.tensor(chain_idx), 22,self.cut_length)
        else:
            interface_residues = np.arange(modeled_idx.shape[0]).tolist()



        # 提取和更新特征
        # extracted_data = {key: value[interface_residues] for key, value in chain_feats.items()}
        #modeled_idx = modeled_idx[interface_residues]

        # 更新残基掩码和链索引
        res_mask = torch.tensor(processed_feats['bb_mask']).int()[interface_residues]
        chain_idx = torch.tensor(chain_idx).int()[interface_residues]
        chain_idx = chain_idx - torch.min(chain_idx) + 1
        try:
            chain_idx = chain_idx - torch.min(chain_idx) + 1
        except:
            print('error ',processed_file_path)


        res_idx = torch.tensor(res_idx).int()[interface_residues]
        res_idx = res_idx - torch.min(res_idx) + 1

        chain_feats={          'atoms14': atoms14[interface_residues],
                               'atoms14_b_factors': atoms14_b_factors[interface_residues],
                               'atoms14_mask': atoms14_mask[interface_residues],
                               'chi': chi[interface_residues],
                               'mask_chi': mask_chi[interface_residues],
                               'res_idx': res_idx,
                               'chain_idx': chain_idx,
                               'res_mask': res_mask,
                               'aatype': aatype[interface_residues],
                               'com_idx': com_idx[interface_residues],
                               'ss': ss[interface_residues],
                               }

        from data.interpolant import save_pdb
        bb=chain_feats['atoms14'][...,:4,:]
        chain_idx=chain_feats['chain_idx']
        #save_pdb(bb.reshape(-1, 3), chain_idx, 'test_interface.pdb')


        if len(chain_idx)>self.cut_length:
            print('too long :',name)

        if len(chain_idx) < self.min_num_res:
            print('too short :', name)
            return None


        feats=[chain_feats]

        if chain_feats['ss'].shape[0] != chain_feats['atoms14'].shape[0] :
            print('error  different length :',name)

        # from data.interpolant import save_pdb
        #
        # save_pdb(bbatoms.detach().cpu().reshape(-1,3),chain_idx,'test.pdb')

        return feats
    def _process_csv_row_noncomplex(self, processed_file_path,idx):
        # 读取并解析处理后的特征
        processed_feats = du.read_pkl(processed_file_path)
        name = processed_file_path.split('/')[-1].split('.')[0]
        print(name)
        # processed_feats = du.read_pkl('//media/junyu/DATA/mmcif/compedidx/f3//1f31.pkl')


        processed_feats = du.parse_chain_feats(processed_feats)

        # 使用布尔索引筛选出有用的原子位置和链索引
        bool_indices = processed_feats['bb_mask'].astype(bool)
        selected_A = processed_feats['atom_positions'][bool_indices][..., [0, 1, 2, 4], :]
        chain_idx = processed_feats['chain_index'][bool_indices]
        modeled_idx = processed_feats['modeled_idx']
        # 删除不再使用的键
        del processed_feats['modeled_idx']


        # if np.all(chain_idx == chain_idx[0]):
        #
        #     print(name,' only one chain')
        #     return []

        # debug


        # print(bool_indices.sum())
        # for i in processed_feats:
        #     print(i)
        #     xx=processed_feats[i][modeled_idx.tolist()]
        #
        # if processed_feats['ss'].shape[0]!=processed_feats['aatype'].shape[0]:
        #     print(name, ' is not right SS ')


        try:

            if np.max(modeled_idx) >= processed_feats['ss'].shape[0]:
                # 给定的数值，我们想删除所有大于这个数值的元素
                threshold = processed_feats['ss'].shape[0]
                # 创建一个新数组，仅包含不大于threshold的元素
                modeled_idx_ss = modeled_idx[modeled_idx <= threshold]

                processed_featsx = tree.map_structure(lambda x: x[modeled_idx_ss.tolist()], processed_feats)
            else:
                processed_featsx = tree.map_structure(lambda x: x[modeled_idx.tolist()], processed_feats)
            processed_feats=processed_featsx
        except:
            try:
                processed_featsy = tree.map_structure(lambda x: x[bool_indices], processed_feats)
                processed_feats = processed_featsy
            except:
                print('map error: ',name)

        # 转换特征为张量并应用映射
        aatype=torch.tensor(processed_feats['aatype']).long()
        com_idx= torch.tensor(processed_feats['com_idx'])



        ss=torch.tensor(vectorized_mapping(processed_feats['ss']))
        print(ss)

        # 获取残基索引和链索引
        res_idx = processed_feats['residue_index']
        chain_idx = processed_feats['chain_index']

        atoms37 = torch.tensor(processed_feats['atom_positions']).float()

        # 获取atoms14相关特征
        residx_atom14_to_atom37, restype_atom14_mask = generate_atoms14(aatype)
        atoms14 = restype_atom14_mask[..., None] * gather_atoms(atoms37, residx_atom14_to_atom37)
        chi, mask_chi = atom14tochis(atoms14.unsqueeze(0), torch.tensor(chain_idx).unsqueeze(0),
                                     aatype.unsqueeze(0))
        chi = chi.squeeze(0)
        mask_chi = mask_chi.squeeze(0)

        atoms14_b_factors = gather_atoms(torch.tensor(processed_feats['b_factors']).unsqueeze(-1),
                                         residx_atom14_to_atom37).squeeze(-1)
        atoms14_mask = gather_atoms(torch.tensor(processed_feats['atom_mask']).unsqueeze(-1),
                                    residx_atom14_to_atom37).squeeze(-1)

        # 根据模型索引处理界面残基

        if modeled_idx.shape[0] > self.cut_length:
            interface_residues, _ = calculate_interface_residues_v3(atoms14, torch.tensor(chain_idx),torch.tensor(chain_idx), 22)

            if len(interface_residues) == 0:
                print(name, ' no interface')
                return []

            interface_residues = sorted(interface_residues)
        else:
            interface_residues = np.arange(modeled_idx.shape[0]).tolist()

        # 提取和更新特征
        # extracted_data = {key: value[interface_residues] for key, value in chain_feats.items()}
        #modeled_idx = modeled_idx[interface_residues]

        # 更新残基掩码和链索引
        res_mask = torch.tensor(processed_feats['bb_mask']).int()[interface_residues]
        chain_idx = torch.tensor(chain_idx).int()[interface_residues]

        try:
            chain_idx = chain_idx - torch.min(chain_idx) + 1
        except:
            print('error ',processed_file_path)


        res_idx = torch.tensor(res_idx).int()[interface_residues]
        res_idx = res_idx - torch.min(res_idx) + 1

        chain_feats={          'atoms14': atoms14[interface_residues],
                               'atoms14_b_factors': atoms14_b_factors[interface_residues],
                               'atoms14_mask': atoms14_mask[interface_residues],
                               'chi': chi[interface_residues],
                               'mask_chi': mask_chi[interface_residues],
                               'res_idx': res_idx,
                               'chain_idx': chain_idx,
                               'res_mask': res_mask,
                               'aatype': aatype[interface_residues],
                               'com_idx': com_idx[interface_residues],
                               'ss': ss[interface_residues],
                               }

        if chain_feats['ss'].shape[0] != chain_feats['atoms14'].shape[0] :
            print('error  different length :',name)


        from data.interpolant import save_pdb
        bb=chain_feats['atoms14'][...,:4,:]
        chain_idx=chain_feats['chain_idx']
        #save_pdb(bb.reshape(-1, 3), chain_idx, 'test_interface_c1.pdb')


        feats=[chain_feats]



        # from data.interpolant import save_pdb
        #
        # save_pdb(bbatoms.detach().cpu().reshape(-1,3),chain_idx,'test.pdb')

        return feats

    def _process_csv_row_complex_nocut(self, processed_file_path,idx):
        # 读取并解析处理后的特征
        processed_feats = du.read_pkl(processed_file_path)
        name = processed_file_path.split('/')[-1].split('.')[0]
        # print(name)
        #processed_feats = du.read_pkl('/media/junyu/DATA/mmcif/compedidx/ep/6epg.pkl')


        processed_feats = du.parse_chain_feats(processed_feats)

        # 使用布尔索引筛选出有用的原子位置和链索引
        bool_indices = processed_feats['bb_mask'].astype(bool)
        selected_A = processed_feats['atom_positions'][bool_indices][..., [0, 1, 2, 4], :]
        chain_idx = processed_feats['chain_index'][bool_indices]
        modeled_idx = processed_feats['modeled_idx']
        # 删除不再使用的键
        del processed_feats['modeled_idx']


        # print(bool_indices.sum())
        # for i in processed_feats:
        #     print(i)
        #     xx=processed_feats[i][modeled_idx.tolist()]
        #
        # if processed_feats['ss'].shape[0]!=processed_feats['aatype'].shape[0]:
        #     print(name, ' is not right SS ')


        try:

            if np.max(modeled_idx) >= processed_feats['ss'].shape[0]:
                # 给定的数值，我们想删除所有大于这个数值的元素
                threshold = processed_feats['ss'].shape[0]
                # 创建一个新数组，仅包含不大于threshold的元素
                modeled_idx_ss = modeled_idx[modeled_idx <= threshold]

                processed_featsx = tree.map_structure(lambda x: x[modeled_idx_ss.tolist()], processed_feats)
            else:
                processed_featsx = tree.map_structure(lambda x: x[modeled_idx.tolist()], processed_feats)
            processed_feats=processed_featsx
        except:
            try:
                processed_featsy = tree.map_structure(lambda x: x[bool_indices], processed_feats)
                processed_feats = processed_featsy
            except:
                print('map error: ',name)

        # 转换特征为张量并应用映射
        # chain_feats = {
        #     'aatype': torch.tensor(processed_feats['aatype']).long(),
        #     'com_idx': torch.tensor(processed_feats['com_idx']),
        #     'ss': torch.tensor(vectorized_mapping(processed_feats['ss']))
        # }
        aatype=torch.tensor(processed_feats['aatype']).long()
        com_idx= torch.tensor(processed_feats['com_idx'])

        # if torch.all(com_idx == com_idx[0]):
        #
        #     print(name,' is not p-p complex')
        #     return []

        ss=torch.tensor(vectorized_mapping(processed_feats['ss']))

        # 获取残基索引和链索引
        res_idx = processed_feats['residue_index']
        chain_idx = processed_feats['chain_index']

        atoms37 = torch.tensor(processed_feats['atom_positions']).float()

        # 获取atoms14相关特征
        residx_atom14_to_atom37, restype_atom14_mask = generate_atoms14(aatype)
        atoms14 = restype_atom14_mask[..., None] * gather_atoms(atoms37, residx_atom14_to_atom37)
        chi, mask_chi = atom14tochis(atoms14.unsqueeze(0), torch.tensor(chain_idx).unsqueeze(0),
                                     aatype.unsqueeze(0))
        chi = chi.squeeze(0)
        mask_chi = mask_chi.squeeze(0)

        atoms14_b_factors = gather_atoms(torch.tensor(processed_feats['b_factors']).unsqueeze(-1),
                                         residx_atom14_to_atom37).squeeze(-1)
        atoms14_mask = gather_atoms(torch.tensor(processed_feats['atom_mask']).unsqueeze(-1),
                                    residx_atom14_to_atom37).squeeze(-1)

        # 根据模型索引处理界面残基
        # in design, we collect all aa, so we need to filter, interface_residues=modeled_idx

        if self._is_training:
            interface_residues, _ = calculate_interface_residues_v3(atoms14, com_idx, torch.tensor(chain_idx), 22,self.cut_length)
        else:
            interface_residues = np.arange(modeled_idx.shape[0]).tolist()



        # 提取和更新特征
        # extracted_data = {key: value[interface_residues] for key, value in chain_feats.items()}
        #modeled_idx = modeled_idx[interface_residues]

        # 更新残基掩码和链索引
        res_mask = torch.tensor(processed_feats['bb_mask']).int()[interface_residues]
        chain_idx = torch.tensor(chain_idx).int()[interface_residues]
        chain_idx = chain_idx - torch.min(chain_idx) + 1
        try:
            chain_idx = chain_idx - torch.min(chain_idx) + 1
        except:
            print('error ',processed_file_path)

        tesx=chain_idx.cpu().numpy()
        res_idx = torch.tensor(res_idx).int()[interface_residues]
        res_idx = res_idx - torch.min(res_idx) + 1

        chain_feats={          'atoms14': atoms14[interface_residues],
                               'atoms14_b_factors': atoms14_b_factors[interface_residues],
                               'atoms14_mask': atoms14_mask[interface_residues],
                               'chi': chi[interface_residues],
                               'mask_chi': mask_chi[interface_residues],
                               'res_idx': res_idx,
                               'chain_idx': chain_idx,
                               'res_mask': res_mask,
                               'aatype': aatype[interface_residues],
                               'com_idx': com_idx[interface_residues],
                               'ss': ss[interface_residues],
                               }

        from data.interpolant import save_pdb
        bb=chain_feats['atoms14'][...,:4,:]
        chain_idx=chain_feats['chain_idx']
        #save_pdb(bb.reshape(-1, 3), chain_idx, 'test_interface.pdb')


        if len(chain_idx)>self.cut_length:
            print('too long :',name)

        if len(chain_idx) < self.min_num_res:
            print('too short :', name)
            return None


        feats=[chain_feats]

        if chain_feats['ss'].shape[0] != chain_feats['atoms14'].shape[0] :
            print('error  different length :',name)

        # from data.interpolant import save_pdb
        #
        # save_pdb(bbatoms.detach().cpu().reshape(-1,3),chain_idx,'test.pdb')

        return feats
    def _process_csv_row(self, processed_file_path,idx):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)
        #processed_feats['atom_positions'] = processed_feats['atom_positions'][..., [0, 1, 2, 4], :]

        bool_indices=processed_feats['bb_mask'].astype(bool)
        # 使用布尔索引从A中选择行
        selected_A = processed_feats['atom_positions'][bool_indices][..., [0, 1, 2, 4], :]
        chain_idx = processed_feats['chain_index'][bool_indices]

        # test map
        # ssnp=processed_feats['ss']
        # ss=vectorized_mapping(processed_feats['ss'])

        from data.interpolant import save_pdb
        #save_pdb(selected_A.reshape(-1, 3), chain_idx, 'test.pdb')

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']

        # maxs=processed_feats['com_idx'].max()
        # if np.max(processed_feats['com_idx']) > 2:
        #     print('com_idx', np.max(processed_feats['com_idx']))
        # return [1]



        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[bool_indices], processed_feats)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']),
            'b_factors': torch.tensor(processed_feats['b_factors']),
            'com_idx': torch.tensor(processed_feats['com_idx']),
            # 应用映射
            'ss':  torch.tensor(vectorized_mapping(processed_feats['ss']))

        }
        # chain_feats = data_transforms.atom37_to_frames(chain_feats)
        # rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        # rotmats_1 = rigids_1.get_rots().get_rot_mats()
        # trans_1 = rigids_1.get_trans()
        res_idx = processed_feats['residue_index']
        chain_idx=processed_feats['chain_index']
        # chain_idx=chain_idx- np.min(chain_idx) + 1
        #
        atoms37=chain_feats['all_atom_positions'].float()

        # get atoms14:
        residx_atom14_to_atom37,restype_atom14_mask=generate_atoms14(chain_feats['aatype'])
        # atoms14=batched_gather(
        #     atoms37,
        #     residx_atom14_to_atom37,
        #     dim=-1,
        #     no_batch_dims=len(atoms37.shape[:-1]),
        # )
        atoms14=restype_atom14_mask[...,None]*gather_atoms(atoms37,residx_atom14_to_atom37)
        chi, mask_chi=atom14tochis(atoms14.unsqueeze(0),torch.tensor(chain_idx).unsqueeze(0),chain_feats['aatype'].unsqueeze(0))
        chi=chi.squeeze(0)
        mask_chi=mask_chi.squeeze(0)
        # reatoms14,remask=chitoatoms(atoms14[...,:4,:].unsqueeze(0),torch.tensor(chain_idx).unsqueeze(0),chain_feats['aatype'].unsqueeze(0),chi)
        # dff=atoms14-reatoms14

        ### dis
        interface_residues,interface_mask=calculate_interface_residues_v2(atoms14,chain_feats['com_idx'],10)

        # 对每个键对应的tensor按给定索引提取值
        extracted_data = {key: value[interface_residues] for key, value in chain_feats.items()}
        modeled_idx=modeled_idx[interface_residues]


        feats = []
        res_mask = torch.tensor(processed_feats['bb_mask']).int()
        chain_idx = torch.tensor(chain_idx).int()
        chain_idx=chain_idx- torch.min(chain_idx) + 1


        res_idx = torch.tensor(res_idx).int()
        res_idx = res_idx - torch.min(res_idx) + 1

        tr=res_idx.numpy()

        if len(chain_idx)>self.cut_length:
            modeled_idx=np.arange(len(modeled_idx))
            sub_modeled_idxs = split_list(modeled_idx, L=self.cut_length)
            # to tensor


            for sub_modeled_idx in sub_modeled_idxs:
                if len(sub_modeled_idx) >= self.dataset_cfg.min_num_res:
                    min_idx = np.min(sub_modeled_idx)
                    max_idx = np.max(sub_modeled_idx)

                    sub_res_idx = res_idx[min_idx:(max_idx + 1)]
                    sub_chain_idx = chain_idx[min_idx:(max_idx + 1)]

                    subf = {
                        'aatype': chain_feats['aatype'][min_idx:(max_idx + 1)],
                        'res_idx': sub_res_idx ,
                        'atoms14': atoms14[min_idx:(max_idx + 1)],
                        'res_mask': res_mask[min_idx:(max_idx + 1)],
                        'chain_idx': sub_chain_idx ,
                        'csv_idx': torch.ones(1, dtype=torch.long) * idx,
                        'b_factors': chain_feats['b_factors'][min_idx:(max_idx + 1)],
                        'ss': chain_feats['ss'][min_idx:(max_idx + 1)],
                        'chi': chi[min_idx:(max_idx + 1)],
                        'mask_chi': mask_chi[min_idx:(max_idx + 1)],
                    }
                    feats.append(subf)


        else:
            min_idx = 0
            max_idx = len(chain_idx) - 1


            res_idx = res_idx[min_idx:(max_idx + 1)]
            chain_idx = chain_idx[min_idx:(max_idx + 1)]
            # chain_idx = chain_idx - torch.min(chain_idx) + 1

            subf = {
                'aatype': chain_feats['aatype'][min_idx:(max_idx + 1)],
                'res_idx': res_idx ,
                'atoms14': atoms14[min_idx:(max_idx + 1)],
                'res_mask': res_mask[min_idx:(max_idx + 1)],
                'chain_idx': chain_idx,
                'csv_idx': torch.ones(1, dtype=torch.long) * idx,
                'b_factors': chain_feats['b_factors'][min_idx:(max_idx + 1)],
                'ss': chain_feats['ss'][min_idx:(max_idx + 1)],
                'chi': chi[min_idx:(max_idx + 1)],
                'mask_chi': mask_chi[min_idx:(max_idx + 1)],
            }
            feats.append(subf)
        # from data.interpolant import save_pdb
        #
        # save_pdb(bbatoms.detach().cpu().reshape(-1,3),chain_idx,'test.pdb')

        return feats
    def _process_csv_row_modeled(self, processed_file_path,idx):
        self.cutmode='cut'
        processed_feats = du.read_pkl(processed_file_path)
        # processed_feats = du.parse_chain_feats(processed_feats)
        #processed_feats['atom_positions'] = processed_feats['atom_positions'][..., [0, 1, 2, 4], :]

        bool_indices=processed_feats['bb_mask'].astype(bool)
        # 使用布尔索引从A中选择行
        #selected_A = processed_feats['atom_positions'][bool_indices]
        chain_idx = processed_feats['chain_index'][bool_indices]

        # from data.interpolant import save_pdb
        # save_pdb(selected_A.reshape(-1, 3), chain_idx, 'test.pdb')

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']


        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[bool_indices], processed_feats)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask'])
        }
        # chain_feats = data_transforms.atom37_to_frames(chain_feats)
        # rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        # rotmats_1 = rigids_1.get_rots().get_rot_mats()
        # trans_1 = rigids_1.get_trans()
        res_idx = processed_feats['residue_index']
        chain_idx=processed_feats['chain_index']
        # chain_idx=chain_idx- np.min(chain_idx) + 1
        #


        feats = []
        res_mask = torch.tensor(processed_feats['bb_mask']).int()
        chain_idx = torch.tensor(chain_idx).int()
        res_idx = torch.tensor(res_idx).int()

        min_idx = 0
        max_idx = len(chain_idx)-1


        res_idx = res_idx[min_idx:(max_idx + 1)]
        chain_idx = chain_idx[min_idx:(max_idx + 1)]
        chain_idx = chain_idx - torch.min(chain_idx) + 1

        bbatoms=chain_feats['all_atom_positions'].float()

        subf = {
            'aatype': chain_feats['aatype'][min_idx:(max_idx + 1)],
            'res_idx': res_idx - torch.min(res_idx) + 1,
            'bbatoms': bbatoms[min_idx:(max_idx + 1)],
            'res_mask': res_mask[min_idx:(max_idx + 1)],
            'chain_idx': chain_idx,
            'csv_idx': torch.ones(1, dtype=torch.long) * idx
        }
        feats.append(subf)
        # from data.interpolant import save_pdb
        #
        # save_pdb(bbatoms.detach().cpu().reshape(-1,3),chain_idx,'test.pdb')

        return feats
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row['processed_path']

        if  self._cut_mode:
            if self.dataset_cfg.complex==1:
                chain_feats = self._process_csv_row_noncomplex(processed_file_path, idx)
            else:
                chain_feats = self._process_csv_row_complex(processed_file_path,idx)
        else:
            #chain_feats = self._process_csv_row_modeled(processed_file_path,idx)
            chain_feats = self._process_csv_row_complex_nocut(processed_file_path, idx)
        return chain_feats
