import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from mad.model.layers.ops.rope import apply_rope
import torch.nn.functional as F
import math


class AddAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        expand_v: int = 1,
        num_heads: int = 1,
        dim_inner = 16,
        use_rope: bool = True,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = int(self.dim * expand_k)
        self.value_dim = int(self.dim * expand_v)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        
        self.dim_inner = dim_inner
        self.use_rope = use_rope

        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        #self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)

        self.fc1 = nn.Linear(1, self.dim_inner)
        self.fc2 = nn.Linear(self.dim_inner, self.dim)
       
        

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):

        b, l = hidden_states.size()[:2]
        # q = self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        # k = self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        # v = self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(torch.bfloat16).contiguous()
        q = (self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        k = (self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        #v = (self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(torch.bfloat16).contiguous()

        if self.use_rope: #note: used to cast q,k,v in bfloat16 AFTER applying rope.
            pos = torch.arange(l, device=hidden_states.device)
            q, k = apply_rope(q, k, positions=pos, base=32.0)

        exp_qk = torch.exp(torch.einsum('bqhd,bkhd->bhqk', q,k) / self.head_qk_dim**(.5))
        exp_qk = exp_qk * torch.tril(torch.ones(1,1,q.size(1), k.size(1), dtype=bool, device=q.device))

        #32x32
        #zero out first 20 rows
        #t=20 is the first of the outputs, attends just to 0 and 10
        #t=21 attends to t=20, 1, 10

        o = torch.sum(exp_qk, dim=-1,keepdims=True).transpose(1,2) #o = torch.einsum('bhqk,bkhd->bqhd', exp_qk, v)

        o = self.fc1(o.flatten(2))
        o = F.relu(o)
        o = self.fc2(o)

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













def ridge_update_W_out(W_in, W_out, k_chunk, v_chunk, lam=1e-3, beta=None):
    """
    W_in:   (b, h, n, d)   memory keys
    W_out:  (b, h, n, dv)  memory values
    k_chunk:(b, t, n, d)   chunk keys (use k as 'query' for memory)
    v_chunk:(b, t, n, dv)  chunk values targets
    """
    b, t, n, d = k_chunk.shape
    h = W_in.shape[1]
    dv = v_chunk.shape[-1]
    if beta is None:
        beta = 1.0 / math.sqrt(d)

    # Compute attention weights A over memory atoms (softmax kernel, bounded in (0,1))
    # scores: (b, n, t, h)
    scores = torch.einsum('btnd,bhnd->bnth', k_chunk.float(), W_in.float()) * beta
    A = torch.softmax(scores, dim=-1)  # softmax over h

    # We want solve for each (b,n):
    # (A^T A + lam I) X = A^T V
    # where A: (t,h), V: (t,dv), X: (h,dv)
    A_ = A.reshape(b*n, t, h)                  # (B, t, h)
    V_ = v_chunk.float().permute(0,2,1,3).reshape(b*n, t, dv)  # (B, t, dv)

    AtA = torch.einsum('bth,btg->bhg', A_, A_)  # (B, h, h)
    AtV = torch.einsum('bth,btd->bhd', A_, V_)  # (B, h, dv)

    # Ridge
    I = torch.eye(h, device=AtA.device, dtype=AtA.dtype).unsqueeze(0)  # (1,h,h)
    M = AtA + lam * I                                                  # (B,h,h)

    # Stable solve (Cholesky is fastest/stablest if PD)
    L = torch.linalg.cholesky(M)                                       # (B,h,h)
    X = torch.cholesky_solve(AtV, L)                                   # (B,h,dv)

    W_out_new = X.reshape(b, n, h, dv).permute(0,2,1,3).to(W_out.dtype)  # (b,h,n,dv)
    return W_out_new