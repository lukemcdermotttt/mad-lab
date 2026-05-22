import torch
import torch.nn as nn
from flash_attn import flash_attn_func
import torch.nn.functional as F
import math

"""
Notes / Ideas to Try:
 - Query Gate to control bandwidth of attention (doesnt work for MLP attention)
 - Low Rank Updates to W_in and W_out.
 - KL Penalty on the update, to keep W_in and W_out from drifting too much.
 - Online Self-Distillation (or why doesn't this work?)
 - Explore Boltzman/Hopfield Style Losses, let (m(q),y) be an optimization problem over y, not neccesarily a closed form solution.
"""

class SimpleMLPAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        expand_v: int = 1,
        num_heads: int = 4,
        dim_inner = 4,
        chunk_size: int = 16,
        base_lr: float = 1e-3,
        use_rope: bool = False,
        use_momentum: bool = False,
        use_weight_decay: bool = False,
        base_weight_decay: float = 0.99,
        base_momentum_decay: float = .9,
        use_nonlinear_projection: bool = False,
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

        if use_nonlinear_projection:
            self.proj_q = nn.Sequential(
                nn.Linear(self.dim, self.key_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.key_dim, self.key_dim, bias=False))
            self.proj_k = nn.Sequential(
                nn.Linear(self.dim, self.key_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.key_dim, self.key_dim, bias=False))
            self.proj_v = nn.Sequential(
                nn.Linear(self.dim, self.value_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.value_dim, self.value_dim, bias=False))
        else:
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
        self.use_momentum = use_momentum
        self.use_weight_decay = use_weight_decay
        self.base_momentum_decay = base_momentum_decay
        self.base_weight_decay = base_weight_decay
        if use_momentum:
            self.proj_momentum_decay = nn.Linear(self.dim, self.num_heads*2, bias=False)
        if use_weight_decay:
            self.proj_weight_decay = nn.Linear(self.dim, self.num_heads*2, bias=False)

        self.gate_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.head_v_dim)


    def forward(self,
        hidden_states: torch.Tensor,
        past_theta = None,
        *args, **kwargs
    ):

        b, l = hidden_states.size()[:2]
        q = self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        k = self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(torch.bfloat16).contiguous()
        v = self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(torch.bfloat16).contiguous()


        if self.use_rope: #note: used to cast q,k,v in bfloat16 AFTER applying rope.
            pos = torch.arange(l, device=hidden_states.device)
            q, k = apply_rope(q, k, positions=pos, base=10000.0)

        lr = torch.nn.functional.softplus(self.proj_lr(hidden_states).float() + self.base_lr).to(q.dtype)
        lr = lr.view(b, l, self.num_heads, 2) #2 for lr_in and lr_out
        if self.use_momentum:
            M_in = torch.zeros_like(self.W_in_init).expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
            M_out = torch.zeros_like(self.W_out_init).expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
            momentum_decay = F.sigmoid(self.proj_momentum_decay(hidden_states).float() + self.base_momentum_decay).to(q.dtype)
            momentum_decay = momentum_decay.view(b, l, self.num_heads, 2)
    
        o = torch.empty_like(v)

        if past_theta is None:
            W_in =  self.W_in_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
            W_out = self.W_out_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
        else:
            W_in, W_out = past_theta

        for i in range(0, l, self.chunk_size):
            s, e = i, min(l, i+self.chunk_size), #max(i-self.chunk_size,0), min(l, i+self.chunk_size)

            """Online Forward Pass""" #(added padding to q with W_in so that q.size(1)==k.size(1), then we get rid of padding after)
            o[:, i:i+self.chunk_size] = flash_attn_func(
                torch.cat([W_in.expand(-1,-1,self.num_heads, -1), q[:,s:e]], dim=1),
                torch.cat([W_in.expand(-1,-1,self.num_heads, -1), k[:, s:e]], dim=1),
                torch.cat([W_out.expand(-1,-1,self.num_heads, -1), v[:, s:e]], dim=1),
                causal=True)[:, -(min(i+self.chunk_size, l)-i):]
            #o[:, i:i+self.chunk_size] = flash_attn_func(q[:, i:i+self.chunk_size], W_in, W_out, causal=False)
            
            """Online Backward Pass""" #(Computing Grad_in and Grad_out each requires flash_attn call, so we instead batch them along head dim)
            lr_in, lr_out = lr[:,s:e,:,:1].mean(dim=1,keepdim=True), lr[:,s:e,:,1:].mean(dim=1,keepdim=True)
            Grad_in, Grad_out = compute_gradients(W_in, W_out, k[:,s:e], v[:,s:e])
            if self.use_momentum:
                momentum_decay_in, momentum_decay_out = momentum_decay[:,s:e,:,:1].mean(dim=1,keepdim=True), momentum_decay[:,s:e,:,1:].mean(dim=1,keepdim=True)
                M_in = M_in * momentum_decay_in + Grad_in * (1-momentum_decay_in)
                M_out = M_out * momentum_decay_out + Grad_out * (1-momentum_decay_out) 
                W_in  = W_in  - lr_in * M_in
                W_out = W_out - lr_out * M_out
            else:
                W_in  = W_in  - lr_in * Grad_in
                W_out = W_out - lr_out * Grad_out

        #Output gating from titans/atlas
        o = self.layer_norm(o)
        o = o.view(b, l, -1)
        out_gate = self.gate_proj(hidden_states)
        o = self.out_proj(o * out_gate)
        #o = self.out_proj(o.view(b, l, -1))

        if past_theta is None:
            return o
        else:
            return o, (W_in, W_out)

    

def compute_gradients(W_in, W_out, k, v, rule="inducing"):
    if rule=="fast_kk":
        W_cat = torch.cat([W_in,  W_in],  dim=2).contiguous()
        K_cat = torch.cat([k, k / 2], dim=2)
        V_cat = torch.cat([v, k], dim=2)
        G = flash_attn_func(W_cat, K_cat, V_cat, causal=False)
        Grad_out, Grad_in = (-G).chunk(2, dim=2)
    elif rule=="fast_vk":
        W_cat = torch.cat([W_in,  W_out],  dim=2).contiguous()
        K_cat = torch.cat([k, v], dim=2)
        V_cat = torch.cat([v, k], dim=2)
        G = flash_attn_func(W_cat, K_cat, V_cat, causal=False)
        Grad_out, Grad_in = (-G).chunk(2, dim=2)
    elif rule=="slow":
        z = torch.einsum('bnhd,bDhd->bnhD', k, W_in)
        p = torch.softmax(z, dim=1)
        Grad_out = torch.einsum('bnhD,bnhd->bDhd', -p, v)

        a = torch.einsum('bDhd,bnhd->bnhD', W_out, v)
        s = (p*a).sum(dim=-1, keepdim=True)
        g_z = -(p*a - p*s)
        Grad_in = torch.einsum('bnhd,bnhD->bDhd', k, g_z)

    elif rule=="inducing":
        W_cat = torch.cat([W_in,  W_out],  dim=3).contiguous()
        KV_cat = torch.cat([k, v], dim=3).contiguous()
        exp_qk = torch.einsum('bDhd,bnhd->bDhn', W_cat, KV_cat).softmax(dim=1)
        Grad = torch.einsum('bDhn,bnhd->bDhd', exp_qk, KV_cat)
        Grad_out, Grad_in = (-Grad).chunk(2, dim=3)


    elif rule=="delta":
        raise NotImplementedError
    elif rule=="hebb": #Proper Hebbian Rule, I think my softmax trick is wrong.
        raise NotImplementedError
    else: 
        raise NotImplementedError
    
    return Grad_in.to(torch.bfloat16), Grad_out.to(torch.bfloat16)




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








def rotate_half(x):
    # x [..., d]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(q, k, *, positions, base=10000.0):
    """
    q, k: (b, t, h, d)
    positions: (t,) or (b, t) integer positions in [0, ...]
    """
    b, t, h, d = q.shape
    assert d % 2 == 0, "RoPE requires head_dim even"

    # Build inverse frequencies
    device = q.device
    dtype = q.dtype
    inv_freq = (base ** (-torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))  # (d/2,)

    # positions -> (t,) in float32
    if positions.dim() == 2:
        # (b, t) -> we’ll broadcast per batch below
        pos = positions.to(torch.float32)  # (b,t)
        freqs = pos[..., None] * inv_freq[None, None, :]  # (b,t,d/2)
        cos = torch.cos(freqs).to(dtype=dtype)  # (b,t,d/2)
        sin = torch.sin(freqs).to(dtype=dtype)
        cos = cos[:, :, None, :].expand(b, t, h, d // 2)  # (b,t,h,d/2)
        sin = sin[:, :, None, :].expand(b, t, h, d // 2)
    else:
        pos = positions.to(device=device, dtype=torch.float32)  # (t,)
        freqs = pos[:, None] * inv_freq[None, :]  # (t,d/2)
        cos = torch.cos(freqs).to(dtype=dtype)[None, :, None, :].expand(b, t, h, d // 2)
        sin = torch.sin(freqs).to(dtype=dtype)[None, :, None, :].expand(b, t, h, d // 2)

    # Interleave cos/sin to full dim via pairwise application
    # q_rot = q * cos_full + rotate_half(q) * sin_full, same for k
    # Build cos_full/sin_full as [..., d] by repeating each element twice
    cos_full = torch.repeat_interleave(cos, repeats=2, dim=-1)  # (b,t,h,d)
    sin_full = torch.repeat_interleave(sin, repeats=2, dim=-1)

    q_out = (q * cos_full) + (rotate_half(q) * sin_full)
    k_out = (k * cos_full) + (rotate_half(k) * sin_full)
    return q_out, k_out



if __name__ == "__main__":
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    

    model = SimpleMLPAttention(
        dim=128,
        num_heads=4,
        dim_inner=256,
        chunk_size=256,
        use_rope=False,
    ).to(device).to(torch.bfloat16).eval()

    # model = Attention(
    #     dim=128,
    #     causal=True,
    #     n_heads=4,
    #     rotary_emb_dim=0.,
    #     dropout=0.0,
    #     window_size=(-1, -1),
    #     num_heads_kv=None,
    #     cross_attn=False,
    #     qkv_proj_bias=True,
    #     out_proj_bias=True,
    #     softmax_scale=None,
    #     dwconv=False,
    #     rotary_emb_base=10000.0,
    #     rotary_emb_scale_base=None,
    #     rotary_emb_interleaved=False,
    #     use_alibi=False,
    #     fused_bias_fc=False,
    #     use_flash_attn=True,
    #     return_residual=False,
    #     device=device,
    #     dtype=torch.bfloat16
    # ).eval()
    chunk_size = 4194304
    seq_lens = [4194304, 8388608, 16777216, 33554432, 67108864]
    bsz = 1

    for L in seq_lens:
        

        # # warmup
        # with torch.inference_mode():
        #     for _ in range(1):
        #         x = torch.randn(bsz, chunk_size, 128, device=device, dtype=torch.bfloat16)
        #         _ = model(x)
        # torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        past_theta = (model.W_in_init.expand(bsz,-1,-1,-1).to(device).to(torch.bfloat16), model.W_out_init.expand(bsz,-1,-1,-1).to(device).to(torch.bfloat16))
        with torch.inference_mode():
            start.record()
            for _ in range(0,L,chunk_size):
                x = torch.randn(bsz, chunk_size, 128, device=device, dtype=torch.bfloat16)
                y, past_theta = model(x, past_theta=past_theta)
            end.record()

        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end)
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        print(f"L={L:5d} | prefill={latency_ms:8.3f} ms | peak_vram={peak_vram_gb:6.3f} GB")