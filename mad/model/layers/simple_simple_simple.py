import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from mad.model.layers.ops.rope import apply_rope
import torch.nn.functional as F
import math

class AttentionPlus(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        expand_v: int = 1,
        num_heads: int = 4,
        **kwargs
    ):
        super().__init__()

        self.dim = dim 
        self.num_heads = num_heads
        self.key_dim = int(self.dim * expand_k)
        self.value_dim = int(self.dim * expand_v)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim)
        self.k_conv1d = ShortConvolution(self.key_dim)
        self.v_conv1d = ShortConvolution(self.value_dim)
        
        self.W_in_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_qk_dim)) / math.sqrt(self.head_qk_dim))
        self.W_out_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_v_dim)) / math.sqrt(self.dim_inner))
        
        self.proj_lr = nn.Linear(self.dim, self.num_heads*2, bias=False)

        self.gate_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.head_v_dim)


    def forward(self,
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):

        #Generate Q, K, V
        b, l = hidden_states.size()[:2]
        q = self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        k = self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        v = self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(torch.bfloat16).contiguous()
        q, k = apply_rope(q, k, positions=torch.arange(l, device=hidden_states.device), base=10000.0)

        o = flash_attn_func(q,k,v, causal=True)

        #Output gating from titans/atlas
        o = self.layer_norm(o)
        o = o.view(b, l, -1)
        out_gate = self.gate_proj(hidden_states)
        o = self.out_proj(o * out_gate)

        return o
    




class ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        x = F.pad(x, (self.kernel_size - 1, 0))  # causal left pad
        y = self.conv(x)
        y = y + x[:, :, self.kernel_size - 1:]  # residual
        y = y.transpose(1, 2)
        return y