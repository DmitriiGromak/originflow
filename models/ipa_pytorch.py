# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified code of Openfold's IPA."""

import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid
from data import all_atom
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d

def calculate_entropy(tensor):
    # 确保张量是三维的
    if tensor.dim() != 3:
        tensor = tensor.squeeze(0)
    assert tensor.dim() == 3, "The tensor must be 3-dimensional."

    # 获取矩阵的行数和列数
    N, L, _ = tensor.shape

    # 对每个切片（即每个LxL矩阵）进行最大最小值归一化
    # 在每个维度上分别计算最小值和最大值
    min_values = tensor.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
    max_values = tensor.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]

    normalized_tensor = (tensor - min_values) / (max_values - min_values)

    # 确保归一化后的张量中的值都在[0, 1]范围内
    normalized_tensor = torch.clamp(normalized_tensor, min=0, max=1)

    # 计算每个元素的概率
    prob_tensor = normalized_tensor / normalized_tensor.sum(dim=(1, 2), keepdim=True)

    # 计算每个元素的对数概率
    log_prob = torch.log(prob_tensor + 1e-10)  # 添加一个小的常数以避免对数为负无穷大

    # 计算信息熵
    entropy = -torch.sum(prob_tensor * log_prob, dim=(1, 2))

    return entropy.mean()  # 对所有样本的平均信息熵


def calculate_entropy_torch_fixed(tensor):
    # 找到非零的概率值
    nonzero_mask = tensor > 0
    # 对非零概率计算信息熵贡献
    entropy_contribution = torch.zeros_like(tensor)
    entropy_contribution[nonzero_mask] = -tensor[nonzero_mask] * torch.log(tensor[nonzero_mask])
    # 对每行的信息熵贡献求和，得到每行的信息熵
    row_entropy = torch.sum(entropy_contribution, dim=2)
    # 计算每个子矩阵的平均信息熵
    average_entropy = torch.mean(row_entropy, dim=1)
    return average_entropy
@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value
def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    ipa_pts = pts[:, :, None, :, :, :] - torch.tile(ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1))
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3),
        ipa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)
    return  phi_angles



class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )



class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if unsqueeze_dim==0:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        cos1 = cos[position_ids]
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)

        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
class LightInvariantPointAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 8,
        pairwise_repr_dim = 32,
        require_pairwise_repr = False,
        eps = 1e-8,
        precision='16-mixed',
            **kwargs
    ):
        super(LightInvariantPointAttention,self).__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr
        self.max_position_embeddings=512
        self.precision = precision
        # NTK rope
        self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
            scalar_key_dim,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=2,
            base=10000,
        )
        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias = False)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias = False)

        # pairwise representation projection to attention bias

        pairwise_repr_dim = default(pairwise_repr_dim, dim) if require_pairwise_repr else 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads),
                Rearrange('b ... h -> (b h) ...')
            )

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)

        self.to_outs = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

    def forward(
        self,
        single_repr,
        pairwise_repr = None,
        *,
        rotations,
        translations,
        position_ids,
        mask = None
    ):
        x, b, h, eps, require_pairwise_repr = single_repr, single_repr.shape[0], self.heads, self.eps, self.require_pairwise_repr
        assert not (require_pairwise_repr and not exists(pairwise_repr)), 'pairwise representation must be given as second argument'



        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways

        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)

        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)



        # split out heads

        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h),
                                           (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h = h, c = 3), (q_point, k_point, v_point))

        rotations = repeat(rotations, 'b n r1 r2 -> (b h) n r1 r2', h = h)
        translations = repeat(translations, 'b n c -> (b h) n () c', h = h)

        # rotate qkv points into global frame

        q_point = einsum('b n d c, b n c r -> b n d r', q_point, rotations) + translations
        k_point = einsum('b n d c, b n c r -> b n d r', k_point, rotations) + translations
        v_point = einsum('b n d c, b n c r -> b n d r', v_point, rotations) + translations

        # compute rotary embeddings and apply rotary position embedding
        kv_seq_len = single_repr.shape[1]

        cos, sin = self.rotary_emb(v_scalar, seq_len=kv_seq_len)

        q_scalar, k_scalar = apply_rotary_pos_emb(q_scalar, k_scalar, cos, sin, position_ids,unsqueeze_dim=2)
        #move here for rope
        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n h d -> b n (h d)', h=h),
                                           (q_scalar, k_scalar, v_scalar))
        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                                           (q_scalar, k_scalar, v_scalar))
        # derive attn logits for scalar and pairwise

        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale


        ###
        scale=math.log(kv_seq_len, self.max_position_embeddings/2)
        scale=torch.tensor(scale, device=single_repr.device)
        attn_logits_scalar=attn_logits_scalar




        if require_pairwise_repr:
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale

        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim = (-1, -2))

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () ()', b = b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale)

        # combine attn logits


        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise

        # mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention
        attn_logits=attn_logits#*scale
        attn = attn_logits.softmax(dim = - 1)
        # attennp=attn.detach().cpu().numpy()[0]

        with disable_tf32(), autocast(enabled = False):
            # disable TF32 for precision

            # aggregate values
            if self.precision=='16-mixed':
                results_scalar = einsum('b i j, b j d -> b i d', attn.half(), v_scalar)
            else:
                results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar) #

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = h)

            if require_pairwise_repr:
                results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values
            if self.precision == '16-mixed':
                results_points = einsum('b i j, b j d c -> b i d c', attn.half(), v_point)  #
            else:
                results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)


            # rotate aggregated point values back into local frame

            results_points = einsum('b n d c, b n c r -> b n d r', results_points - translations, rotations.transpose(-1, -2))
            results_points_norm = torch.sqrt( torch.square(results_points).sum(dim=-1) + eps )

        # merge back heads

        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)',b=b, h = h)



        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', b=b,h = h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)',b=b, h = h)

        results = (results_scalar, results_points, results_points_norm)

        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)',b=b, h = h)
            results = (*results, results_pairwise)

        # concat results and project out

        results = torch.cat(results, dim = -1)
        return self.to_outs(results)
class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        self.max_position_embeddings=512
        # NTK rope
        self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
            self.c_hidden,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=2,
            base=10000,
        )

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()


        kv_seq_len = k.shape[1]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids,unsqueeze_dim=2)
        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        # e_qk= calculate_entropy(a.squeeze(0))
        length=s.shape[1]
        scale=math.log(length, 256)
        scale=torch.tensor(scale, device=s.device)


        #e_qk_scale=calculate_entropy(a.squeeze(0))

        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        #e_b=calculate_entropy(permute_final_dims(b, (2, 0, 1)))


        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        # e_pt=calculate_entropy(pt_att)


        a = a + pt_att
        a = a * scale
        #e_all=calculate_entropy(a)

        a = a + square_mask.unsqueeze(-3)
        # a = a * scale
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )
        
        return s,None#[e_qk, e_qk_scale,e_b, e_pt, e_all]


class SInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(SInvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf
        print('ipadim: ',ipa_conf.c_s,'c_hidden: ',ipa_conf.c_hidden)
        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        self.max_position_embeddings = 512
        # NTK rope
        self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
            self.c_hidden,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=2,
            base=10000,
        )

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)
        #self.c_z // 4 +
        concat_out_dim = (
                 self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="normal")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,
            position_ids: torch.Tensor,
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # # [*, N_res, N_res, H]
        # b = self.linear_b(z[0])

        if (_offload_inference):
            z[0] = z[0].cpu()

        kv_seq_len = k.shape[0]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2)
        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        # e_qk= calculate_entropy(a.squeeze(0))
        length = s.shape[1]
        scale = math.log(length, self.max_position_embeddings / 2)
        scale = torch.tensor(scale, device=s.device)
        # if length<self.max_position_embeddings / 2:
        #     scale=torch.ones_like(scale)


        # e_qk_scale=calculate_entropy(a.squeeze(0))

        #a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))



        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        # e_pt=calculate_entropy(pt_att)

        a = a + pt_att
        # a = a * scale
        # e_all=calculate_entropy(a)

        a = a + square_mask.unsqueeze(-3)
        a = a * scale
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # if (_offload_inference):
        #     z[0] = z[0].to(o_pt.device)

        # # [*, N_res, H, C_z // 4]
        # pair_z = self.down_z(z[0])
        # o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        #
        # # [*, N_res, H * C_z // 4]
        # o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )

        return s, None  # [e_qk, e_qk_scale,e_b, e_pt, e_all]
class TorsionAngles(nn.Module):
    def __init__(self, c, num_torsions, eps=1e-8):
        super(TorsionAngles, self).__init__()

        self.c = c
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        # TODO: Remove after published checkpoint is updated without these weights.
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom

        return unnormalized_s, normalized_s


class RotationVFLayer(nn.Module):
    def __init__(self, dim):
        super(RotationVFLayer, self).__init__()

        self.linear_1 = Linear(dim, dim, init="relu")
        self.linear_2 = Linear(dim, dim, init="relu")
        self.linear_3 = Linear(dim, dim)
        self.final_linear = Linear(dim, 6, init="final")
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return self.final_linear(s)


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, use_rot_updates):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self._use_rot_updates = use_rot_updates
        update_dim = 6 if use_rot_updates else 3
        self.linear = Linear(self.c_s, update_dim, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update

class IpaScore(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(IpaScore, self).__init__()
        self._model_conf = model_conf
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.diffuser = diffuser

        self.scale_pos = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / ipa_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()

        for b in range(ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(ipa_conf.c_s)
            self.trunk[f'skip_embed_{b}'] = Linear(
                self._model_conf.node_embed_size,
                self._ipa_conf.c_skip,
                init="final"
            )
            tfmr_in = ipa_conf.c_s + self._ipa_conf.c_skip
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, ipa_conf.seq_tfmr_num_layers)
            self.trunk[f'post_tfmr_{b}'] = Linear(
                tfmr_in, ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                c=ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(ipa_conf.c_s)

            if b < ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                    node_embed_size=ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.torsion_pred = TorsionAngles(ipa_conf.c_s, 1)

    def forward(self, init_node_embed, edge_embed, input_feats):
        node_mask = input_feats['res_mask'].type(torch.float32)
        diffuse_mask = (1 - input_feats['fixed_mask'].type(torch.float32)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        init_frames = input_feats['rigids_t'].type(torch.float32)

        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)
        init_rots = init_rigids.get_rots()

        # Main trunk
        curr_rigids = self.scale_rigids(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_in = torch.cat([
                node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
            ], dim=-1)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                seq_tfmr_in, src_key_padding_mask=1 - node_mask)
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * diffuse_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, diffuse_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        rot_score = self.diffuser.calc_rot_score(
            init_rigids.get_rots(),
            curr_rigids.get_rots(),
            input_feats['t']
        )
        rot_score = rot_score * node_mask[..., None]

        curr_rigids = self.unscale_rigids(curr_rigids)
        trans_score = self.diffuser.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            input_feats['t'][:, None, None],
            use_torch=True,
        )
        trans_score = trans_score * node_mask[..., None]
        _, psi_pred = self.torsion_pred(node_embed)
        model_out = {
            'psi': psi_pred,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'final_rigids': curr_rigids,
        }
        return model_out
