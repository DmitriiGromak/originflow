"""Neural network for embedding node features."""
import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding
import numpy as np
import data.residue_constants as rc
def aa_psy_che(indexfile='/AAindex1',lookup = None):

    title='A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V'
    oldindex=title.split()
    index=[]
    for i in oldindex:
        index.append(i[0])
    for i in oldindex:
        index.append(i[2])


    with open(indexfile) as f:
        contents = f.readlines()
    n=0
    data=[]
    for i in range(len(contents)):
        this_proper=[]

        if contents[i][:1]=='I':
            n=n+1
            for a in list(contents[i + 1].split('\n')[0].split()):
                if a == 'NA':
                    a = 0.
                this_proper.append(float(a) )
            for a in list(contents[i + 2].split('\n')[0].split()):
                if a=='NA':
                    a=0.

                this_proper.append(float(a))

            data.append(this_proper)

    npdata=torch.as_tensor(np.asarray(data),dtype=torch.float)
    npx=torch.mean(npdata,-1).unsqueeze(-1)
    digits=torch.transpose(torch.cat((npdata,npx),-1),0,1)
    index.append('X')

    new_list=[]
    for i in list(lookup):
        aa=index.index(i)
        new_list.append(digits[aa])

    digits=torch.stack((new_list),0)
    return digits
def one_hot_encode_b_factors_with_overflow(b_factors, max_value=400, interval=40):
    """
    将B因子的值域按照给定的区间宽度进行独热编码，并处理溢出值。

    参数:
        b_factors (torch.Tensor): 一个包含B因子的张量，形状为[B,N]。
        max_value (int): B因子的最大值域，用于确定编码的最大区间。
        interval (int): 划分区间的宽度。

    返回:
        torch.Tensor: 独热编码的B因子，形状为[N, num_intervals+1]，其中num_intervals为总的区间数，额外的一列用于溢出值。
    """
    # 确定总的区间数
    num_intervals = max_value // interval

    # 计算每个B因子所在的区间索引
    interval_indices = (b_factors / interval).floor().long()

    # 对于超过最大值域的B因子，将其区间索引设置为最后一个区间（即溢出类别）
    interval_indices[interval_indices >= num_intervals] = num_intervals

    # # 初始化独热编码张量，额外的一列用于溢出值
    # one_hot_encoded = torch.zeros(b_factors.size(0), num_intervals + 1, dtype=torch.float32)
    #
    # # 填充独热编码
    # one_hot_encoded.scatter_(1, interval_indices.unsqueeze(1), 1)

    return interval_indices


class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg,mode):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb

        self.mode=mode
        if self.mode=='motif':

            self.c_fixedmask_emb=self._cfg.c_fixedmask_emb
            self.linears = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_fixedmask_emb, self.c_s)

        if self.mode=='fbb':

            self.c_fixedmask_emb=self._cfg.c_fixedmask_emb
            self.linears = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_fixedmask_emb, self.c_s)



        elif self.mode=='base':
            # we fine from motif still use this, so no need to add c_fixedmask_emb
            # self.linear = nn.Linear(
            #     self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)
            self.c_fixedmask_emb = self._cfg.c_fixedmask_emb
            self.linears = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_fixedmask_emb, self.c_s)


        elif self.mode=='base_ss':
            # we fine from motif still use this, so no need to add c_fixedmask_emb
            self.SS_embed = nn.Embedding(num_embeddings=4, embedding_dim=self._cfg.c_SS_emb)
            self.c_fixedmask_emb = self._cfg.c_fixedmask_emb
            self.linear_ss = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_SS_emb+self._cfg.c_fixedmask_emb, self.c_s)


    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask,fixed_mask,ss=None):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(
            pos, self.c_pos_emb, max_len=2056
        )
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # native show worse
        # nativepos_emb = get_index_embedding(
        #     nativepos, self.c_pos_emb, max_len=10000
        # )
        # nativepos_emb = nativepos_emb * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [pos_emb]
        # timesteps are between 0 and 1. Convert to integers.

        # time embed
        time_embed=self.embed_t(timesteps, mask)
        if fixed_mask is not None:
            # make time embed for motif area
            motif_time=torch.ones_like(timesteps)
            motif_time_embed = self.embed_t(motif_time, mask)

            time_embed=time_embed*(1-fixed_mask.unsqueeze(-1))+motif_time_embed*fixed_mask.unsqueeze(-1)
            input_feats.append(time_embed)
            input_feats.append(get_index_embedding(fixed_mask, self.c_fixedmask_emb, max_len=2056))
        else:

            input_feats.append(time_embed)
            if self.mode != 'base':
                input_feats.append(get_index_embedding(mask, self.c_fixedmask_emb, max_len=2056))

        if self.mode=='motif' or self.mode=='fbb':
            return self.linears(torch.cat(input_feats, dim=-1))
        if self.mode=='base':

            #0414 finetune from motif
            input_feats.append(get_index_embedding(mask, self.c_fixedmask_emb, max_len=2056))
            return self.linears(torch.cat(input_feats, dim=-1))

        if self.mode == 'base_ss':
            # 0414 finetune from motif

            SS_emb = self.SS_embed(ss)
            input_feats.append(SS_emb)
            return self.linear_ss(torch.cat(input_feats, dim=-1))
            # viognn
            # return self.linear(torch.cat(input_feats, dim=-1))


class NodeEmbedder_v2(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder_v2, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.c_fixedmask_emb=self._cfg.c_fixedmask_emb


        self.SS_embed = nn.Embedding(num_embeddings=4, embedding_dim=self._cfg.c_SS_emb)


        ### sequence
        self.Seq_embed = nn.Embedding(num_embeddings=21, embedding_dim=self._cfg.c_AA_emb)
        alphabet = ''.join(rc.restypes_with_x)
        aatype_pc_embed = aa_psy_che(indexfile='../data/AAindex1', lookup=alphabet)
        self.register_buffer("aatype_pc_embed", aatype_pc_embed)
        self.Seq_linear = nn.Linear(in_features=self._cfg.c_AA_emb+self.aatype_pc_embed.shape[-1], out_features=self._cfg.c_AA_emb, bias=False)

        # chi
        self.chi_embed = nn.Linear(in_features=8, out_features=self._cfg.c_Chi_emb, bias=False)

        # bfacotr
        self.trunk = nn.ModuleDict()
        self.bfactor_embed_interval=[15,30,45,60]
        self.trunk[f'bfactor_embed_native'] = nn.Linear(in_features=1, out_features=64, bias=False)
        for th in self.bfactor_embed_interval:
            self.trunk[f'bfactor_embed_{th}'] = nn.Embedding(num_embeddings=int(self._cfg.max_bf//th)+1, embedding_dim=64)
        self.trunk[f'bfactor_embed_linear'] = nn.Linear(in_features=64*(len(self.bfactor_embed_interval)+1), out_features=self._cfg.c_bfactor_emb, bias=False)


        # useless
        self.linears = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_timestep_emb, self.c_s)
        # useless

        self.linear = nn.Linear(
            self._cfg.c_pos_emb +
            self._cfg.c_timestep_emb+
            self._cfg.c_SS_emb+
            self._cfg.c_AA_emb +  #seq
            self._cfg.c_Chi_emb +  #chi
            self._cfg.c_bfactor_emb +  #bf
            self._cfg.c_fixedmask_emb, self.c_s)
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)
    def embed_chi(self, chi,mask_chi):
        """
        将χ角转换为正弦和余弦值。

        参数:
            chi_angles (torch.Tensor): 一个包含χ角的张量，其形状为[B, N, 4]，值范围为-pi到pi。

        返回:
            sin_features (torch.Tensor): χ角的正弦值，形状为[B, N, 4]。
            cos_features (torch.Tensor): χ角的余弦值，形状为[B, N, 4]。
        """
        chi_emb = torch.cat([torch.sin(chi)*mask_chi, torch.cos(chi)*mask_chi], dim=2)
        return self.chi_embed(chi_emb)

    def embed_bfactor(self,factor):
        bf_emb=[]
        bf_emb.append(self.trunk[f'bfactor_embed_native'](factor.unsqueeze(-1)))
        for th in self.bfactor_embed_interval:
            bf_emb.append(
                self.trunk[f'bfactor_embed_{th}'](
                    one_hot_encode_b_factors_with_overflow(factor,self._cfg.max_bf,interval=th))
            )
        bf_emb=torch.cat(bf_emb,dim=-1)
        bf_emb=self.trunk[f'bfactor_embed_linear'](bf_emb)
        return bf_emb

    def embed_seq(self,seq):
        aa_emb=self.Seq_embed(seq)
        chem_f = self.aatype_pc_embed[seq]
        return  self.Seq_linear(torch.cat([aa_emb,chem_f],dim=-1))

    def forward(self, timesteps,mask,is_training,fixed_mask,res_idx,ss,aatype,chi,mask_chi,atoms14_b_factors ,**kw_args):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(res_idx, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # SS we do not let mask area = 0
        SS_emb=self.SS_embed(ss)
        # Seq
        Seq_emb=self.embed_seq(aatype)

        # chi
        chi_emb=self.embed_chi(chi,mask_chi)

        # bf
        b_factor=atoms14_b_factors[...,:,1]
        bf_emb=self.embed_bfactor(b_factor)


        if is_training:
            # [b, n_res, c_timestep_emb]
            input_feats = [pos_emb,SS_emb,Seq_emb*fixed_mask[...,None],chi_emb*fixed_mask[...,None],bf_emb*fixed_mask[...,None]]
            # timesteps are between 0 and 1. Convert to integers.
        else:
            # [b, n_res, c_timestep_emb]
            input_feats = [pos_emb,SS_emb,Seq_emb,chi_emb,bf_emb]

        # time embed
        time_embed=self.embed_t(timesteps, mask)
        if fixed_mask is not None:
            # make time embed for motif area, the finial time is 1
            motif_time=torch.ones_like(timesteps)
            motif_time_embed = self.embed_t(motif_time, mask)

            time_embed=time_embed*(1-fixed_mask.unsqueeze(-1))+motif_time_embed*fixed_mask.unsqueeze(-1)
            input_feats.append(time_embed)
            input_feats.append(get_index_embedding(fixed_mask, self.c_fixedmask_emb, max_len=2))  # finxed area and generate area
        else:

            input_feats.append(time_embed)
            input_feats.append(get_index_embedding(mask, self.c_fixedmask_emb, max_len=2))
        return self.linear(torch.cat(input_feats, dim=-1))



    def forward_forfbb(self,timesteps, mask,is_training,fixed_mask,res_idx,ss,aatype,chi,mask_chi,atoms14_b_factors ,**kw_args):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(res_idx, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # SS we do not let mask area = 0
        SS_emb=self.SS_embed(ss)
        # Seq
        Seq_emb=self.embed_seq(aatype)

        # chi
        chi_emb=self.embed_chi(chi,mask_chi)

        # bf
        b_factor=atoms14_b_factors[...,:,1]
        bf_emb=self.embed_bfactor(b_factor)


        if is_training:
            # [b, n_res, c_timestep_emb]
            input_feats = [pos_emb,SS_emb,Seq_emb*fixed_mask[...,None],chi_emb*fixed_mask[...,None],bf_emb*fixed_mask[...,None]]
            # timesteps are between 0 and 1. Convert to integers.
        else:
            # [b, n_res, c_timestep_emb]
            input_feats = [pos_emb,SS_emb,Seq_emb,chi_emb,bf_emb]


        # time embed
        time_embed=self.embed_t(timesteps, mask)
        if fixed_mask is not None:
            # make time embed for motif area, the finial time is 1
            motif_time=torch.ones_like(timesteps)
            motif_time_embed = self.embed_t(motif_time, mask)

            time_embed=time_embed*(1-fixed_mask.unsqueeze(-1))+motif_time_embed*fixed_mask.unsqueeze(-1)
            input_feats.append(time_embed)
            input_feats.append(get_index_embedding(fixed_mask, self.c_fixedmask_emb, max_len=2))  # finxed area and generate area
        else:

            input_feats.append(time_embed)
            input_feats.append(get_index_embedding(mask, self.c_fixedmask_emb, max_len=2))
        return self.linear(torch.cat(input_feats, dim=-1))

class NodeEmbedder_v3(nn.Module):
    '''

    add base with ss

    '''

    def __init__(self, module_cfg,mode):
        super(NodeEmbedder_v3, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb

        self.SS_embed = nn.Embedding(num_embeddings=4, embedding_dim=self._cfg.c_SS_emb)

        self.mode=mode
        if self.mode=='motif':

            self.c_fixedmask_emb=self._cfg.c_fixedmask_emb
            self.linears = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_fixedmask_emb, self.c_s)

        if self.mode=='fbb':

            self.c_fixedmask_emb=self._cfg.c_fixedmask_emb
            self.linears = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_fixedmask_emb, self.c_s)



        elif self.mode=='base':
            # we fine from motif still use this, so no need to add c_fixedmask_emb
            # self.linear = nn.Linear(
            #     self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)
            self.c_fixedmask_emb = self._cfg.c_fixedmask_emb
            self.linears = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_fixedmask_emb, self.c_s)


        elif self.mode=='base_ss':
            # we fine from motif still use this, so no need to add c_fixedmask_emb
            # self.linear = nn.Linear(
            #     self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)
            self.c_fixedmask_emb = self._cfg.c_fixedmask_emb
            self.linears = nn.Linear(
                self._cfg.c_pos_emb + self._cfg.c_timestep_emb + self._cfg.c_SS_emb+self._cfg.c_fixedmask_emb, self.c_s)


    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, timesteps, mask,fixed_mask,ss):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(
            pos, self.c_pos_emb, max_len=2056
        )
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # SS we do not let mask area = 0
        SS_emb=self.SS_embed(ss)

        # [b, n_res, c_timestep_emb]
        input_feats = [pos_emb]
        # timesteps are between 0 and 1. Convert to integers.

        # time embed
        time_embed=self.embed_t(timesteps, mask)
        if fixed_mask is not None:
            # make time embed for motif area
            motif_time=torch.ones_like(timesteps)
            motif_time_embed = self.embed_t(motif_time, mask)

            time_embed=time_embed*(1-fixed_mask.unsqueeze(-1))+motif_time_embed*fixed_mask.unsqueeze(-1)
            input_feats.append(time_embed)
            input_feats.append(get_index_embedding(fixed_mask, self.c_fixedmask_emb, max_len=2056))
        else:

            input_feats.append(time_embed)
            if self.mode != 'base':
                input_feats.append(get_index_embedding(mask, self.c_fixedmask_emb, max_len=2056))

        if self.mode=='motif' or self.mode=='fbb':
            return self.linears(torch.cat(input_feats, dim=-1))
        if self.mode=='base':

            #0414 finetune from motif
            input_feats.append(get_index_embedding(mask, self.c_fixedmask_emb, max_len=2056))
            return self.linears(torch.cat(input_feats, dim=-1))

        if self.mode == 'base_ss':
            # 0414 finetune from motif
            input_feats.append(get_index_embedding(mask, self.c_fixedmask_emb, max_len=2056))
            return self.linears(torch.cat(input_feats, dim=-1))
            # viognn
            # return self.linear(torch.cat(input_feats, dim=-1))

# if __name__ == '__main__':
#     # 示例
#
#     ss=torch.randint(0,21,(1,128))
#     ssembd=x[ss]
#     print(x)

