
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mad.model.layers.ops.rope import apply_rope

class GaussianAttention(nn.Module):

    def __init__(
        self,
        dim: int = 1024,
        expand_v: float = 2.0,
        expand_k: float = 1.0,
        num_heads: int = 4,
        layernorm_eps: float = 1e-5,
        *args, **kwargs
    ) -> GaussianAttention:
        super().__init__()
        self.d_model = dim
        self.value_dim = int(self.d_model * expand_v)
        self.key_dim = int(self.d_model * expand_k)
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.num_heads = num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, self.d_model, bias=False)
        self.eps = 1e-5

        self.reset_parameters()
        self.use_rope = True

        # concat(sort(slice(x,0,find(x,0))),single(0))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)

    def forward(self, x):
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        
        b,n = x.size(0), x.size(1)
        #QK Norm
        #q = F.normalize(q, dim=-1)
        #k = F.normalize(k, dim=-1)

        
        if self.use_rope:
            pos = torch.arange(n, device=x.device)  # (l,)
            q, k = apply_rope(q.transpose(1,2), k.transpose(1,2), positions=pos, base=10000.0)
            q, k = q.transpose(1,2), k.transpose(1,2)
                
        exp_qk = torch.tril(torch.exp(torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.head_qk_dim**(-.5))) # b h nq nk
        exp_kk = torch.tril(torch.exp(torch.einsum('b h i d, b h j d -> b h i j', k, k) * self.head_qk_dim**(-.5)))  # b h nk nk

        
        eye = torch.eye(n, device=x.device, dtype=exp_kk.dtype).expand(b, self.num_heads, n, n)
        #set torch.diagonal(exp_kk) to 1 everywhere.
        
        #Alternative to inverse for stability
        #inv_kk = torch.linalg.inv(exp_kk.float()+eye*self.eps).to(x.dtype)
        #o = torch.tril(exp_qk) @ inv_kk @ v
        
        o = exp_qk.float() @ torch.linalg.solve_triangular(exp_kk.float() + eye.float() * self.eps, v.float(), upper=False)
        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.o_proj(o.to(x.dtype))

        return o