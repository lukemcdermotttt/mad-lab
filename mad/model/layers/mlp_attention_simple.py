import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from mad.model.layers.ops.rope import apply_rope
import torch.nn.functional as F
import math


class SimpleMLPAttention(nn.Module):
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
            s, e = i, min(l, i+self.chunk_size), #max(i-self.chunk_size,0), min(l, i+self.chunk_size)
            #Online Forward Pass (added padding to q with W_in so that q.size(1)==k.size(1), then we get rid of padding after)
            o[:, i:i+self.chunk_size] = flash_attn_func(
                torch.cat([W_in, q[:,s:e]], dim=1),
                torch.cat([W_in, k[:, s:e]], dim=1),
                torch.cat([W_out, v[:, s:e]], dim=1),
                causal=True)[:, -(min(i+self.chunk_size, l)-i):]
            
            #o[:, i:i+self.chunk_size] = flash_attn_func(q[:, i:i+self.chunk_size], W_in, W_out, causal=False)
            
            #Online Backward Pass - (Computing Grad_in and Grad_out each requires flash_attn call, so we instead batch them along head dim)
            if i < l - self.chunk_size: #Don't update on last chunk since, since its meaningless overhead.
                Grad_in, Grad_out = updates(W_in, W_out, k[:,s:e], v[:,s:e])
                W_in  = W_in  - lr[:,s:e,:,:1].mean(dim=1,keepdim=True) * Grad_in
                W_out = W_out - lr[:,s:e,:,1:].mean(dim=1,keepdim=True) * Grad_out

        #Output gating from titans/atlas
        o = self.layer_norm(o)
        o = o.view(b, l, -1)
        out_gate = self.gate_proj(hidden_states)
        o = self.out_proj(o * out_gate)
        #o = self.out_proj(o.view(b, l, -1))
        return o
    


def updates(W_in, W_out, k, v, rule="fast_kk"):
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
    elif rule=="delta":
        raise NotImplementedError
    elif rule=="hebb": #Proper Hebbian Rule, I think my softmax trick is wrong.
        raise NotImplementedError
    else: 
        raise NotImplementedError

    return Grad_in.to(torch.bfloat16), Grad_out.to(torch.bfloat16)





#KL Penalty for the update?
#for one feature vecotr, W_i
#KL(new||old)= \sum_j exp(W k_j) 
#maybe we plug in k instead of q
#P(v_i|q) \log(P(v_i|q) / Q(v_i|q))
#P(v_i|q) = \exp(q^T W_i)
#Q(v_i|q) = \exp(q^T W_i_old)
#P(v_i|q) / Q(v_i|q) = \exp(q^T (W_i - W_i_old))
#sum_i \exp(q^T W_i) * q^T (W_i - W_i_old) for i in dim_inner.






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









"""CODE USED TO MEASURE LATENCY AND VRAM USAGE, NOT PART OF THE MODEL DEFINITION"""
# import typing as tp
# from flash_attn.modules.mha import MHA


# class Attention(MHA):
#     """Wrapper for the Multi-Head Attention module from the `flash_attn` package."""
#     def __init__(self,
#         dim: int,
#         causal: bool = True,
#         n_heads: int = 16,
#         rotary_emb_dim: float = 0.,
#         dropout: float = 0.0,
#         window_size: tp.Tuple[int, int] = (-1, -1),
#         num_heads_kv: int = None,
#         cross_attn: bool = False,
#         qkv_proj_bias: bool = True,
#         out_proj_bias: bool = True,
#         softmax_scale: float = None,
#         dwconv: bool = False,
#         rotary_emb_base: float = 10000.0,
#         rotary_emb_scale_base: float = None,
#         rotary_emb_interleaved: bool = False,
#         use_alibi: bool = False,
#         fused_bias_fc: bool = False,
#         use_flash_attn: bool = True,
#         return_residual: bool = False,
#         device=None,
#         dtype=None,
#         *args, **kwargs
#     ) -> None:
#         super().__init__(
#             embed_dim=dim,
#             num_heads=n_heads,
#             rotary_emb_dim=rotary_emb_dim,
#             dropout=dropout,
#             causal=causal,
#             window_size=window_size,
#             use_flash_attn=use_flash_attn,
#             num_heads_kv=num_heads_kv,
#             cross_attn=cross_attn,
#             qkv_proj_bias=qkv_proj_bias,
#             out_proj_bias=out_proj_bias,
#             softmax_scale=softmax_scale,
#             dwconv=dwconv,
#             rotary_emb_base=rotary_emb_base,
#             rotary_emb_scale_base=rotary_emb_scale_base,
#             rotary_emb_interleaved=rotary_emb_interleaved,
#             use_alibi=use_alibi,
#             fused_bias_fc=fused_bias_fc,
#             return_residual=return_residual,
#             device=device,
#             dtype=dtype,
#         )


# if __name__ == "__main__":
#     device = "cuda"
#     torch.backends.cuda.matmul.allow_tf32 = True
    

#     model = SimpleMLPAttention(
#         dim=128,
#         num_heads=4,
#         dim_inner=256,
#         chunk_size=256,
#         use_rope=False,
#     ).to(device).to(torch.bfloat16).eval()

#     # model = Attention(
#     #     dim=128,
#     #     causal=True,
#     #     n_heads=4,
#     #     rotary_emb_dim=0.,
#     #     dropout=0.0,
#     #     window_size=(-1, -1),
#     #     num_heads_kv=None,
#     #     cross_attn=False,
#     #     qkv_proj_bias=True,
#     #     out_proj_bias=True,
#     #     softmax_scale=None,
#     #     dwconv=False,
#     #     rotary_emb_base=10000.0,
#     #     rotary_emb_scale_base=None,
#     #     rotary_emb_interleaved=False,
#     #     use_alibi=False,
#     #     fused_bias_fc=False,
#     #     use_flash_attn=True,
#     #     return_residual=False,
#     #     device=device,
#     #     dtype=torch.bfloat16
#     # ).eval()

#     seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864]
#     bsz = 1

#     for L in seq_lens:
#         x = torch.randn(bsz, L, 128, device=device, dtype=torch.bfloat16)

#         # # warmup
#         with torch.inference_mode():
#             for _ in range(3):
#                 _ = model(x)
#         torch.cuda.synchronize()

#         torch.cuda.reset_peak_memory_stats()
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)

#         with torch.inference_mode():
#             start.record()
#             _ = model(x)
#             end.record()

#         torch.cuda.synchronize()
#         latency_ms = start.elapsed_time(end)
#         peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

#         print(f"L={L:5d} | prefill={latency_ms:8.3f} ms | peak_vram={peak_vram_gb:6.3f} GB")