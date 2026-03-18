import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from mad.model.layers.ops.rope import apply_rope
import torch.nn.functional as F
import math
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

class Semilinear(nn.Module):
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
        self.proj_lr = nn.Linear(self.dim, self.num_heads*2, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim)
        self.k_conv1d = ShortConvolution(self.key_dim)
        self.v_conv1d = ShortConvolution(self.value_dim)
        self.W_in_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_qk_dim)) / math.sqrt(self.head_qk_dim))
        self.W_out_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_v_dim)) / math.sqrt(self.dim_inner))
        
        

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):

        b, l = hidden_states.size()[:2]
        q = self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        k = self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        v = self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(torch.bfloat16).contiguous()

        if self.use_rope: #note: used to cast q,k,v in bfloat16 AFTER applying rope.
            pos = torch.arange(l, device=hidden_states.device)
            q, k = apply_rope(q, k, positions=pos, base=10000.0)

        lr = self.proj_lr(hidden_states)
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr).to(q.dtype)
        lr = lr.view(b, l, self.num_heads, 2) #2 for lr_in and lr_out, these values tend to sit around [.6,1.2].
    
        o = torch.empty_like(v)
        W_in =  self.W_in_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
        W_out = self.W_out_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)

        for i in range(0, l, self.chunk_size):
            
            k_h = F.softmax(torch.einsum('blhd,bDhd->blhD', k[:, i:i+self.chunk_size], W_in), dim=-1) * lr[:, i:i+self.chunk_size, :, 1:]
            q_h = F.softmax(torch.einsum('blhd,bDhd->blhD', q[:, i:i+self.chunk_size], W_in), dim=-1)
            qk = torch.einsum('bqhD,bkhD->bhqk', q_h, k_h).masked_fill_(torch.triu(torch.ones(1,1,q_h.size(1), q_h.size(1), dtype=bool, device=q.device), diagonal=1), 0)

            #o[:, i:i+self.chunk_size], _ = fused_chunk_linear_attn(q_h, k_h, v[:, i:i+self.chunk_size], 
            #                                scale=None, initial_state=W_out.transpose(1,2), normalize=False)
            o[:, i:i+self.chunk_size] = torch.einsum('bqhD,bDhd->bqhd', q_h, W_out) #initial state prediciton
            o[:, i:i+self.chunk_size] += torch.einsum('bhqk,bkhd->bqhd', qk, v[:, i:i+self.chunk_size]) 

            W_out += torch.einsum('bnhD,bnhd->bDhd', k_h, v[:, i:i+self.chunk_size])  #optionally, use the linear attention update first.

            #Online Backward Pass - (Computing Grad_in and Grad_out each requires flash_attn call, so we instead batch them along head dim)
            if l > self.chunk_size: 
                for _ in range(2):
                    W_cat = torch.cat([W_in,  W_out],  dim=2) 
                    K_cat = torch.cat([k[:, i:i+self.chunk_size], v[:, i:i+self.chunk_size]], dim=2)
                    V_cat = torch.cat([v[:, i:i+self.chunk_size], k[:, i:i+self.chunk_size]], dim=2)
                    G = flash_attn_func(W_cat, K_cat, V_cat, causal=False)
                    Grad_out, Grad_in = (-G).chunk(2, dim=2)

                    W_in  = W_in  - lr[:, i:i+1, :, :1] * Grad_in
                    W_out = W_out - lr[:, i:i+1, :, 1:] * Grad_out

                #v_pred = flash_attn_func(k[:, i:i+self.chunk_size], W_in, W_out, causal=False)
                #loss = -torch.einsum('bnhd,bnhd->bnh', v[:, i:i+self.chunk_size], v_pred)
                #print(torch.mean(loss).item())

        o = self.out_proj(o.view(b, l, -1))
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