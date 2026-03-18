import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from mad.model.layers.ops.rope import apply_rope
import torch.nn.functional as F
import math
#from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn

"""
My attempt on recreating atlas, not using muon for now.

M(x) = x + W2@GeLU(W1@x)


"""

class Atlas(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        dim_inner = 4,
        chunk_size: int = 16,
        base_lr: float = 1e-3,
        use_rope: bool = False,
        act: str = 'gelu',
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = int(self.dim )
        self.value_dim = int(self.dim)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        
        self.dim_inner = dim_inner # #degree 2 polynomial mapping #dim_inner
        self.chunk_size = chunk_size
        self.base_lr = base_lr
        self.use_rope = use_rope

        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_lr = nn.Linear(self.dim, self.num_heads*2, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.gate_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim)
        self.k_conv1d = ShortConvolution(self.key_dim)
        self.v_conv1d = ShortConvolution(self.value_dim)
        self.W_in_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_qk_dim)) / math.sqrt(self.head_qk_dim))
        self.W_out_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_v_dim)) / math.sqrt(self.dim_inner))
        self.layer_norm = nn.LayerNorm(self.head_v_dim)
        

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):

        b, l = hidden_states.size()[:2]
        q = torch.norm(F.SiLU(self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim)),dim=-1, p=2).to(torch.bfloat16).contiguous()
        k = torch.norm(F.SiLU(self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim)),dim=-1, p=2).to(torch.bfloat16).contiguous()
        v = F.SiLU(self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim)).to(torch.bfloat16).contiguous()

        lr = self.proj_lr(hidden_states)
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr).to(q.dtype)
        lr = lr.view(b, l, self.num_heads, 2) #2 for lr_in and lr_out, these values tend to sit around [.6,1.2].
    
        o = torch.empty_like(v)
        W_in =  self.W_in_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
        W_out = self.W_out_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
        #M
        for i in range(0, l, self.chunk_size):
            #W_init q_t + (sum vk) q_t
            #o[:, i:i+self.chunk_size] = 







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
    


def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def d_gelu(x):
    return x * (torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)) + 0.5 * (1 + torch.erf(x / math.sqrt(2)))

"""
My attempt on Titans Linear
Currently uses hebbian loss -<m(k),v> instead of delta for implementation simplicity.
"""

class TitansLinear(nn.Module):
    def __init__(
        self,
        dim: int,
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
        self.key_dim = int(self.dim )
        self.value_dim = int(self.dim)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        
        self.dim_inner = dim_inner # #degree 2 polynomial mapping #dim_inner
        self.chunk_size = chunk_size
        self.base_lr = base_lr
        self.use_rope = use_rope

        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_lr = nn.Linear(self.dim, self.num_heads*1, bias=False)
        self.proj_decay = nn.Linear(self.dim, self.num_heads*1, bias=False)
        self.proj_momentum = nn.Linear(self.dim, self.num_heads*1, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.gate_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim)
        self.k_conv1d = ShortConvolution(self.key_dim)
        self.v_conv1d = ShortConvolution(self.value_dim)
        self.W_init = nn.Parameter(torch.randn((1,self.num_heads,self.head_qk_dim, self.head_v_dim)) / math.sqrt(self.head_qk_dim))
        self.layer_norm = nn.LayerNorm(self.head_v_dim)
        

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):

        b, l = hidden_states.size()[:2]
        q = torch.norm(F.SiLU(self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim)),dim=-1, p=2).to(torch.bfloat16).contiguous()
        k = torch.norm(F.SiLU(self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim)),dim=-1, p=2).to(torch.bfloat16).contiguous()
        v = F.SiLU(self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim)).to(torch.bfloat16).contiguous()

        lr = self.proj_lr(hidden_states)
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr).to(q.dtype)
        lr = lr.view(b, l, self.num_heads, 1) #2 for lr_in and lr_out, these values tend to sit around [.6,1.2].

        momentum = F.SiLU(self.proj_momentum(hidden_states)).view(b, l, self.num_heads, 1)
        weight_decay = F.SiLU(self.proj_decay(hidden_states)).view(b, l, self.num_heads, 1)
    
        o = torch.empty_like(v)
        W = self.W_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
        M = torch.zeros_like(W) #momentum term

        for i in range(0, l, self.chunk_size):
            q_c, k_c, v_c = q[:, i:i+self.chunk_size], k[:, i:i+self.chunk_size], v[:, i:i+self.chunk_size]
            mom_c = momentum[:, i:i+self.chunk_size]
            decay_c = torch.cumprod(weight_decay[:, i:i+self.chunk_size],dim=1)
            

            #o = (W_init + sum vk) q = W_init q_t + (sum vk) q_t = W_init@q + (sum qkv)
            #o = Long-Term Memory(q) + Short-Term Memory(q)
            qk_c = torch.einsum('bqhd,bkhd->bhqk',q_c,k_c)
            qk_c = qk_c * torch.tril(torch.ones(q_c.size(1), k_c.size(1)))[None,None,:,:] #Casual Mask (b,h,q_l,k_l)
            
            #u = torch.einsum('blhk,blhv->blhkv', k_c,v_c)

            o[:, i:i+self.chunk_size] = torch.matmul(q_c, W) + torch.matmul(qk_c, v_c.transpose(1,2))

            W = W + torch.einsum('blhk,blhv->bhkv', k_c,v_c)

        
        o = self.layer_norm(o)
        o = o.view(b, l, -1)
        out_gate = self.gate_proj(hidden_states)
        o = self.out_proj(o * out_gate)
        return o
    
