import glob
import os
import torch
import torch.nn as nn
import hydra
import tqdm
import re
from chroma.data.protein import Protein
from typing import Optional
from data.interpolant import Interpolant_10
from models.flow_model import FlowModel_binder, FlowModel_binder_sidechain,FlowModel_seqdesign
from data import utils as du
from data.pdb_dataloader import sPdbDataset
import numpy as np
import random

class Proflow(nn.Module):

    def __init__(
        self,
        cfg,
        weights_backbone: str = "named:public",
        weights_design: str = "named:public",
        device: Optional[str] = None,
        strict: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        import warnings

        warnings.filterwarnings("ignore")

        # If no device is explicity specified automatically set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self._exp_cfg=cfg.experiment
        self._inf_cfg=cfg.inference

        self._sidechain = FlowModel_binder_sidechain(cfg.model).to(self.device)

        self.Proflow = Interpolant_10(cfg.interpolant)
        self._sidechain.eval()



    def _load_from_state_dict_forsidechain(self, ) -> None:
        print('load from state dict')
        checkpoint = torch.load( self._inf_cfg.sidechain_path)
        original_state_dict = checkpoint['state_dict']
        # 加载权重之前，先调整键名
        adjusted_state_dict = {k[len("model."):]: v for k, v in original_state_dict.items() if k.startswith("model.")}

        # 创建更新后的模型实例
        # 修改原始状态字典以适应新模型（如果需要）
        # 例如，删除不再存在的层的权重或重命名某些键
        # updated_state_dict = {key: value for key, value in adjusted_state_dict.items() if
        #                       key in self._model.state_dict()}

        # 加载匹配的权重到新模型
        self._sidechain.load_state_dict(adjusted_state_dict, strict=True)
        print('side chain load from state dict finished')





    def sample_seqside(self,Target=0,design_num=1):
        '''
        Target=0 is A chain

        '''

        pdb_path='/home/junyu/project/monomer_test/base_neigh/base_rcsbcluster30_fixtopo_motif_1000_fix_update_generateall/2024-06-02_00-02-49/last/rcsb/4zyp_motif/'
        pklpath=f'{pdb_path}/preprocessed/'

        model_name=self._exp_cfg.warm_start.split('/')[-3]

        design_name='4zyp_motif_0'
        design_path=f'{pdb_path}/seqside/'
        ref_path=f'{pklpath}/4zyp_motif_seq.pkl'
        # 检查文件是否存在
        if os.path.exists(ref_path):
            # 如果文件存在，执行的操作
            print("文件存在。")
        else:
            from data.collect_pkl import colletdata
            ref_path=colletdata()

        folder_path=design_path+'seqs/'
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(folder_path)
            print(f"文件夹'{folder_path}'已创建。")
        else:
            # 如果文件夹已存在
            print(f"文件夹'{folder_path}'已存在。")

        pkl=glob.glob(f'{pklpath}*.pkl')

        for i in pkl:

            ref_data = du.read_pkl(i)

            thisname=i.split('/')[-1].split('.')[0]
            # PdbDataset=sPdbDataset(i, is_training=False)
            #
            #
            # for ref_data in PdbDataset:
            #     # ref_data = du.read_pkl(ref_pdb)
            #     # 为每个Tensor增加一个空白维度
            #     del ref_data['csv_idx']
            #     ref_data = {key: value.unsqueeze(0).to(self.device) for key, value in ref_data.items()}
            #
            #     com_idx=ref_data['com_idx']
            #     ref_data['bbatoms']=ref_data['atoms4']
            #     binder_mask=torch.ones_like(com_idx) #all fixed
            #     binder_mask=binder_mask.to(self.device)
            #
            #     self.Proflow.set_device(self.device)
                # noisy_batch = self.Proflow.corrupt_seq(ref_data,)


                # noisy_batch = self.Proflow.corrupt_batch_binder_sidechain(ref_data, 'fp32', t=0.975, noise=True)
                #
                # noisy_batch['atoms14_b_factors']=noisy_batch['atoms14_b_factors'].float()
                # # sample str and seq and same time
                # samples =self._sidechain(noisy_batch)



                # C = ref_data['chain_idx']
                # S = samples['SEQ']


                #design side
            ref_data=dict(ref_data)
            X=torch.tensor(ref_data['atom_positions'])[...,[0,1,2,4],:].unsqueeze(0)
            C= torch.tensor(ref_data['chain_index']).unsqueeze(0)
            S=torch.tensor(ref_data['aatype']).unsqueeze(0)

            p = Protein.from_XCS(X, C, S)
            p.to_PDB(folder_path
            +thisname+'.pdb')

def make_fixed_mask(input_str):
    # 初始化mask列表
    mask = []

    # 分割输入字符串为各个区间
    intervals = re.split(r',\s*', input_str)

    # 处理每个区间
    for interval in intervals:
        if re.match(r'[A-Za-z]', interval):
            # 处理保留区间
            match = re.search(r'(\d+)-(\d+)', interval)
            start, end = map(int, match.groups())
            length = end - start + 1
            mask.extend([1] * length)
        else:
            # 处理随机区间
            start, end = map(int, interval.split('-'))
            length = random.randint(start, end)
            mask.extend([0] * length)

    # 转换为Tensor
    mask_tensor = torch.tensor(mask, dtype=torch.int)

    return mask_tensor


def create_mask(L, indices):
    # 初始化一个长度为 L 的列表，所有元素为 0
    mask = [0] * L

    # 根据提供的索引列表，将相应的元素设置为 1
    for index in indices:
        if 0 <= index < L:  # 确保索引在列表长度范围内
            mask[index] = 1

    return mask
# 解析并处理输入字符串
def process_input(input_str, pdb_dict):
    intervals = re.split(r',\s*', input_str)
    all_positions = []
    # 初始化mask列表
    mask = []
    aatype=[]
    indices_mask=np.zeros_like(pdb_dict['modeled_idx'])

    for interval in intervals:
        if re.match(r'[A-Za-z]', interval):
            # 处理特定链上的残基区间

            chain_char, start, end = re.findall(r'([A-Za-z])(\d+)-(\d+)', interval)[0]
            start, end = int(start), int(end)
            chain_int = du.chain_str_to_int(chain_char)

            # 从pdb_dict中提取相应的原子坐标
            indices = np.where((pdb_dict['chain_index'] == chain_int) &
                               (pdb_dict['residue_index'] >= start) &
                               (pdb_dict['residue_index'] <= end))[0]
            # 使用np.take而不是直接索引，以避免TypeError
            positions = np.take(pdb_dict['atom_positions'], indices, axis=0)
            all_positions.append(positions)
            indices_mask[indices]=1
            seqs_motif=np.take(pdb_dict['aatype'], indices, axis=0)
            aatype.append(seqs_motif)

            length = end - start + 1
            mask.extend([1] * length)
        else:
            # 处理随机采样区间

            length = int(interval)
            # 生成随机坐标，假设每个残基有4个原子，每个原子有3个坐标维度
            random_positions = np.random.rand(length, 37, 3)
            all_positions.append(random_positions)
            aatype.append(np.zeros(length))

            mask.extend([0] * length)
    # 转换为Tensor
    mask_tensor = torch.tensor(mask, dtype=torch.int)

    #indices_mask=create_mask(L=pdb_dict['residue_index'].shape[0],indices=indices_mask)
    return np.concatenate(all_positions,axis=0),mask_tensor,np.concatenate(aatype,axis=0),indices_mask

def test_process_input():
    str='E400-510/31,A24-42,A24-42,A64-82,4'
    ref_pdb=''
    ref_data = du.read_pkl(ref_pdb)

@hydra.main(version_base=None, config_path="../configs", config_name="inference_seq")
def main(cfg):
    # make_fixed_mask()
    # 使用cfg对象


    proflow = Proflow(cfg)
    a = 60  # 起始值
    b = 1000  # 结束值，不包括在内
    n = 20  # 间隔
    sequence = list(range(a, b, n))
    sample_length=sequence
    # proflow.sample_motif()
    # for i in range(50):
    #     proflow.sample_binder(Target=2, design_num=i)

    proflow.sample_seqside(Target=0, design_num=1)

    # for L in [20, 25,30,35,40,45,50,55,60]:
    #     for i in range(5):
    #         proflow.sample_binder_bylength(Target=2,design_num=i,Length=L)


if __name__ == '__main__':


    # import re



    main()


