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
        num_heads: int = 16,
        use_rope: bool = False,
        **kwargs
    ):
        super().__init__()

        self.dim = dim 
        self.num_heads = num_heads
        self.key_dim = int(self.dim * expand_k)
        self.value_dim = int(self.dim * expand_v)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
    
        self.use_rope = use_rope

        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_bandwidth = nn.Linear(self.dim, self.num_heads, bias=False)

        self.gate_proj = nn.Linear(self.dim, self.value_dim, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim)
        self.k_conv1d = ShortConvolution(self.key_dim)
        self.v_conv1d = ShortConvolution(self.value_dim)
        self.layer_norm = nn.LayerNorm(self.head_v_dim)

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):

        b, l = hidden_states.size()[:2]
        q = self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        k = self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        v = self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(torch.bfloat16).contiguous()
        #q_gate = F.tanh(self.proj_bandwidth(hidden_states).unsqueeze(-1))+1 # gate in between [-1,1] per head.

        #QK Norm

        #q = F.normalize(q, dim=-1).to(torch.bfloat16) * torch.pow(math.sqrt(self.head_qk_dim), q_gate).to(torch.bfloat16)
        #k = F.normalize(k, dim=-1).to(torch.bfloat16)

        #q = q * torch.pow(torch.norm(q, dim=-1, keepdim=True), q_gate).to(torch.bfloat16)
        #q = q * torch.pow(math.sqrt(self.head_qk_dim), q_gate).to(torch.bfloat16)

        if self.use_rope: #note: used to cast q,k,v in bfloat16 AFTER applying rope.
            pos = torch.arange(l, device=hidden_states.device)
            q, k = apply_rope(q, k, positions=pos, base=10000.0)

        o = flash_attn_func(q,k,v, causal=True)
        #o = causal_epanechnikov_attention(q, k, v)
        #o = flash_attn_func(q,k.roll(1, dims=1),k, causal=True) #Next-Latent Attention: learn m(k_{t-1}) \approx k_{t}
        

        #Output gating from titans/atlas
        o = self.layer_norm(o)
        o = o.view(b, l, -1)
        out_gate = self.gate_proj(hidden_states)
        o = self.out_proj(o * out_gate)
        #o = self.out_proj(o.view(b, l, -1))
        return o
    


def causal_epanechnikov_attention(q, k, v, bandwidth=None, eps=1e-8):
    if True:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

    if bandwidth is None:
        bandwidth = 1 #2 * q.size(-1) ** 0.5

    qk = torch.einsum('bqhd,bkhd->bhqk', q, k)
    w = F.relu(1.0 - (2.0 - 2.0 * qk) / (bandwidth ** 2))
    mask = torch.tril(torch.ones(q.size(1), q.size(1), device=q.device, dtype=torch.bool))
    w = w.masked_fill(~mask, 0.0)
    w = w / (w.sum(dim=-1, keepdim=True) + eps)

    return torch.einsum('bhqk,bkhd->bqhd', w, v)




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




