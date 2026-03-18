# import torch
# import torch.nn as nn

# class MonarchAttention(nn.Module):
#     def __init__(self, dim: int, num_blocks: int = 16, decay: float = 1.00, clip: float = 1000.0, coef: float = 1, **kwargs):
#         super().__init__()
#         assert dim % num_blocks == 0
#         self.dim = dim
#         self.nb = num_blocks
#         self.db = dim // num_blocks
#         self.decay = decay
#         self.clip = clip
#         self.coef = coef
#         self.proj_k = nn.Linear(dim, dim, bias=False)
#         self.proj_v = nn.Linear(dim, dim, bias=False)
#         self.proj_beta = nn.Linear(dim, num_blocks, bias=False)
#         self.out_proj = nn.Linear(dim, dim, bias=False)
#         perm = torch.arange(dim).view(num_blocks, self.db).t().reshape(-1)
#         self.register_buffer("perm", perm, persistent=False)
#         self.L_state = None
#         self.R_state = None

#     def _ensure_state(self, batch: int, device):
#         if self.L_state is None or self.L_state.size(0) != batch:
#             eye = torch.eye(self.db, device=device)
#             self.L_state = torch.zeros(batch, self.nb, self.db, self.db, device=device)
#             self.R_state = eye.unsqueeze(0).unsqueeze(0).repeat(batch, self.nb, 1, 1)

#     def _seg(self, x):
#         return x.view(*x.shape[:-1], self.nb, self.db)

#     def _P(self, x):
#         return x.index_select(-1, self.perm)

#     def _apply_s(self, L, R, k):
#         rk = torch.matmul(R, self._seg(k).unsqueeze(-1)).squeeze(-1)
#         y = self._P(rk.flatten(-2, -1))
#         ly = torch.matmul(L, self._seg(y).unsqueeze(-1)).squeeze(-1)
#         return self._P(ly.flatten(-2, -1))

#     @torch.no_grad()
#     def _update_state(self, k, v, beta):
#         k_seg = self._seg(k).unsqueeze(-1)
#         w = self._P(torch.matmul(self.R_state, k_seg).squeeze(-1).flatten(-2, -1))
#         w_seg = self._seg(w)
#         pv = self._P(v)
#         e = torch.matmul(self.L_state, w_seg.unsqueeze(-1)).squeeze(-1).flatten(-2, -1) - pv
#         e_seg = self._seg(e)
#         scale = beta.unsqueeze(-1).unsqueeze(-1) * self.coef
#         delta_L = torch.matmul(e_seg.unsqueeze(-1), w_seg.unsqueeze(-2))
#         c = self._P(
#             torch.matmul(
#                 self.L_state.transpose(-1, -2), e_seg.unsqueeze(-1)
#             ).squeeze(-1).flatten(-2, -1)
#         )
#         c_seg = self._seg(c)
#         delta_R = torch.matmul(c_seg.unsqueeze(-1), k_seg.transpose(-2, -1))
#         new_L = (self.L_state * self.decay - scale * delta_L).clamp(-self.clip, self.clip)
#         new_R = (self.R_state * self.decay - scale * delta_R).clamp(-self.clip, self.clip)
#         self.L_state = new_L.detach()
#         self.R_state = new_R.detach()

#     def forward(self, x):
#         b, l, _ = x.shape
#         self._ensure_state(b, x.device)
#         outputs = []
#         for t in range(l):
#             k_t = self.proj_k(x[:, t])
#             v_t = self.proj_v(x[:, t])
#             beta_t = torch.sigmoid(self.proj_beta(x[:, t]))
#             y_hat = self._apply_s(self.L_state.detach(), self.R_state.detach(), k_t)
#             outputs.append(self.out_proj(y_hat))
#             self._update_state(k_t, v_t, beta_t.detach())
#         return torch.stack(outputs, dim=1)

# if __name__ == "__main__":
#     torch.manual_seed(0)
#     x = torch.randn(2, 8, 64, requires_grad=True)
#     attn = MonarchAttention(64, 8)
#     y = attn(x)
#     y.mean().backward()
#     print(torch.isfinite(x.grad).all())


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



class MonarchAttention(nn.Module):
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
        self.proj_k1 = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v1 = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_k2 = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v2 = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_beta1 = nn.Linear(self.dim, self.num_heads , bias=False)
        self.proj_beta2 = nn.Linear(self.dim, self.num_heads , bias=False)
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
        q, k1, k2, v1, v2 = self.proj_q(hidden_states), self.proj_k1(hidden_states), self.proj_k2(hidden_states), self.proj_v1(hidden_states), self.proj_v2(hidden_states)
        q = q.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2)
        k1 = k1.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2)
        k2 = k2.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2)
        v1 = v1.view(b, l, self.num_heads, self.head_v_dim).transpose(1, 2)
        v2 = v2.view(b, l, self.num_heads, self.head_v_dim).transpose(1, 2)


        q, k1, k2 = self.feature_map_q(q), self.feature_map_k(k1), self.feature_map_k(k2)
        if self.norm_q:
            q = q / (q.sum(-1, keepdim=True) + 1e-4)
        if self.norm_k:
            k1 = k1 / (k1.sum(-1, keepdim=True) + 1e-4)
            k2 = k2 / (k2.sum(-1, keepdim=True) + 1e-4)

        y1 = self.parallel_forward(hidden_states, q, k1, v1)
        #y1 = rearrange(y1, 'b h l d -> b l (h d)')
        #o = self.out_proj(y1)
        
        y1 = rearrange(y1, 'b h1 l (h2 d) -> b h1 l h2 d', h2=self.num_heads)
        y1 = rearrange(y1, 'b h1 l h2 d -> b h2 l (h1 d)')
        y2 = self.parallel_forward(hidden_states, y1, k2, v2)
        y2 = rearrange(y2, 'b h2 l d -> b l (h2 d)')
        o = self.out_proj(y2)
        return o

    def parallel_forward(self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ):
        if self.parallel_implementation == "quadratic" or causal_dot_product is None:
            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
            A_qk = torch.tril(A_qk)        
            y = torch.einsum("bhnm,bhme->bhne", A_qk.to(x.dtype), v.to(x.dtype))
            z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
            y = y * z[..., None]
            #y = rearrange(y, 'b h l d -> b l (h d)')
        
        elif self.parallel_implementation == "linear" and causal_dot_product is not None:
            v = causal_dot_product(
                q.contiguous().to(dtype=torch.float32),
                k.contiguous().to(dtype=torch.float32),
                v.contiguous().to(dtype=torch.float32)
            )
            z = 1 / (
                torch.einsum(
                    "bhld,bhld->bhl", 
                    q.to(dtype=torch.float32), 
                    k.to(dtype=torch.float32).cumsum(2)
                ) + self.eps
            )
            y = v * z[..., None]
            #y = rearrange(y, 'b h l d -> b l (h d)')
        
        else: 
            raise ValueError(f"Parallel implementation {self.parallel_implementation} not supported")

        return y.to(x.dtype)

    
class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False, activation: str = 'silu', use_norm: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,      # depthwise
            bias=bias
        )
        #self.act = nn.SiLU()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))  # causal left pad
        y = self.conv(x)
        y = y + x[:, :, self.kernel_size - 1:]  # residual
        #y = self.act(y)
        if self.use_norm:
            y = self.norm(y.transpose(1, 2))
        else:
            y = y.transpose(1, 2)

        return y, None  # back to (B, T, C)
    