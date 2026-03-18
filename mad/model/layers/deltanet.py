# adpated from:
# https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/linear_attn.py
# https://github.com/HazyResearch/based/blob/main/based/models/mixers/linear_attention.py

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
#from fla.ops.delta_rule import chunk_delta_rule

from mad.model.layers.featurization.feature_map import (
    DPFPFeatureMap,
    HadamardFeatureMap,
    HedgehogFeatureMap,
    T2RFeatureMap,
    TaylorFeatureMap
)

try:
    from mad.model.layers.ops.causal_dot_prod import causal_dot_product  # linear attention cuda kernel
except ImportError:
    print(f"causal_dot_product not installed, using quadratic linear attention implementation!... ")
    causal_dot_product = None



class DeltaNet(nn.Module):
    def __init__(
        self,
        dim: int,
        feature_map: str='silu', 
        expand_k: int = 1,
        expand_v: int = 1,
        tie_feature_map_qk: bool = False,
        num_heads: int = 16,
        eps: float = 1e-12,
        parallel_implementation: str="quadratic",
        norm_q: bool = False,
        norm_k: bool = False,
        **kwargs
    ):
        super().__init__()

        assert feature_map in [
            'elu',
            'relu',
            'taylor',
            'hedgehog',
            't2r',
            'dpfp',
            'identity',
            'elementwise_product',
            'silu',
        ], f"Not supported feature map `{feature_map}`."
        
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = int(self.dim * expand_k)
        self.value_dim = int(self.dim * expand_v)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.eps = eps
        self.parallel_implementation = parallel_implementation
        self.assign_feature_map(
            feature_map=feature_map,
            tie_feature_map_qk=tie_feature_map_qk
        )

        # initialize projections and feature map
        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_beta = nn.Linear(self.dim, self.num_heads , bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k

    def assign_feature_map(self, feature_map: str, tie_feature_map_qk: bool = False):
        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'taylor':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = TaylorFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = TaylorFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = TaylorFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        elif feature_map == 'silu':
            self.feature_map_q = nn.SiLU()
            self.feature_map_k = nn.SiLU()
        else:
            raise NotImplementedError

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        b, l, _ = hidden_states.size()
        q, k, v, beta = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states), self.proj_beta(hidden_states)
        q = q.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_v_dim).transpose(1, 2)
        beta = beta.view(b, l, self.num_heads)
            
        q, k = self.feature_map_q(q), self.feature_map_k(k)
        if self.norm_q:
            q = q / (q.sum(-1, keepdim=True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, keepdim=True) + 1e-4)
        
        o, recurrent_state = chunk_delta_rule(q, k, v, beta)
        return  o #self.parallel_forward(hidden_states, q, k, v)
    
    def parallel_forward(self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ):
        if self.parallel_implementation == "fla" or causal_dot_product is None:
            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
            A_qk = torch.tril(A_qk)        
            y = torch.einsum("bhnm,bhme->bhne", A_qk.to(x.dtype), v.to(x.dtype))
            z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
            y = y * z[..., None]
            y = rearrange(y, 'b h l d -> b l (h d)')
        else: 
            raise ValueError(f"Parallel implementation {self.parallel_implementation} not supported")

        return self.out_proj(y.to(x.dtype))



if __name__ == '__main__':
    import torch
    x = torch.randn(2, 128, 128).to(torch.bfloat16).cuda().requires_grad_(True)
    for fm in [
        'elu',
        'relu',
        'hedgehog',
        'taylor',
        't2r',
        'dpfp',
        'identity',
        'elementwise_product'
    ]:
        print(f'Testing deltanet forward with {fm} feature map...')
        model = DeltaNet(dim=128, feature_map=fm).to(torch.bfloat16).cuda()
        y = model(x)
        print(y.shape)
        y.sum().backward()
        print(x.grad.shape)