import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from mad.model.layers.ops.rope import apply_rope
import torch.nn.functional as F
import math
import os
#from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn

"""
This approach uses a 2-Layer MLP in a chunkwise training fashion. 
W_2 (the output layer) is updated every time step with W_1 frozen, similar to linear recurrences. 
Then every c-time steps, both W_1 and W_2 are jointly optimized with backprop.


Todo:
- learning rate in the linear attention loop.
- momentum
- weight decay
- possibly other activations instead of softmax

If fla ops are implemented, we can use their hardcoded backwards to get the gradient of our sigma(1 x) then compute grad of W_1 ourselves, no matter what the fan_out operation is.
We can also use the final_hidden_state from fla ops to be our "gradient update" for W_1 stuff. For example lets try a mesa layer.

Lets also do an experiment where we take one mad-lab task that our mlp-attention does well on (i.e. fuzzy-recall for some combo of vs,seqlen,etc.). 
In our forward pass, lets save the queries,keys,values (and other parameters like beta, W_in_init, W_out_init, etc.) in a dictionary
then we can use a notebook to see how well our operations fit that key-to-value map. We can also save key/values generated from other models too.
This is importatn becasue I imagine the MLP isnt well trained, yet using 2 backrpop steps hurts performance.

If 32 heads works okay (4 key/value dim), then we can actually visualize the keys and values, and see how the query gets mapped / interpolated. Each q,k,v would have the first 3 dimensions be plotted on a 3D plot then have the 4th dimension be the color.
"""

class RL(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        expand_v: int = 1,
        num_heads: int = 16,
        dim_inner = 4,
        chunk_size: int = 16,
        base_lr: float = 1e-3,
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
        
        self.dim_inner = dim_inner
        self.chunk_size = chunk_size
        self.base_lr = base_lr
        self.use_rope = use_rope

        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.proj_lr = nn.Linear(self.dim, self.num_heads, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim)
        self.k_conv1d = ShortConvolution(self.key_dim)
        self.v_conv1d = ShortConvolution(self.value_dim)
        
        self.gate_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.head_v_dim)
        

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):

        b, l = hidden_states.size()[:2]
        q = self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        k = self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        v = self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(torch.bfloat16).contiguous()

        lr = self.proj_lr(hidden_states)
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr).to(q.dtype)
        lr = lr.view(b, l, self.num_heads) 
        
        if self.use_rope: #note: used to cast q,k,v in bfloat16 AFTER applying rope.
            pos = torch.arange(l, device=hidden_states.device)
            q, k = apply_rope(q, k, positions=pos, base=10000.0)
        
        q, k = F.relu(q), F.relu(k)

        W = torch.zeros((b,self.num_heads, self.head_qk_dim, self.head_v_dim), device=q.device, dtype=torch.bfloat16)
        H = torch.zeros((b,self.num_heads, self.head_qk_dim, self.head_v_dim), device=q.device, dtype=torch.bfloat16)
        s = torch.zeros((b,self.num_heads, self.head_qk_dim), device=q.device, dtype=torch.bfloat16)
        o = torch.zeros_like(v)

        
        for i in range(l):
            start, end = i, i+1
            H = H + torch.einsum('blhk,blhv->bhkv', k[:,i:i+1], v[:,i:i+1]) #H_t = H_{t-1} + k_t v_t^T
            s = s + k[:,i] #s_t = s_{t-1} + k_t
            
            T = torch.einsum('blhk,bhkv->blhv', q[:,i:i+1], H)
            Y = torch.einsum('blhk,bhkv->blhv', q[:,i:i+1], W)
            alpha = torch.einsum('blhk,bhk->blh', q[:,i:i+1], s).unsqueeze(-1)

            #FIX START
            s_norm = s.abs().sum(-1, keepdim=True).unsqueeze(1)         # (b,1,h,1)
            alpha = alpha / (s_norm + 1e-8)

            D = T - alpha * Y

            #DEN FIX START
            den = (q[:,i:i+1].square().sum(-1, keepdim=True) + 1e-6)  # (b,1,h,1)
            W_grad = torch.einsum('blhk,blhv->bhkv', q[:,i:i+1], D/den)

            W = W - lr[:,i,:,None,None]* W_grad 

            o[:,i:i+1] = torch.einsum('blhk,bhkv->blhv', q[:,i:i+1], W)
            #print(torch.norm(H), torch.norm(s), torch.norm(T), torch.norm(Y), torch.norm(W), torch.norm(D), torch.norm(W_grad))

        

        #Output gating from titans/atlas
        o = self.layer_norm(o)
        o = o.view(b, l, -1)
        out_gate = self.gate_proj(hidden_states)
        o = self.out_proj(o * out_gate)
        #o = self.out_proj(o.view(b, l, -1))
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