import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.mesa_net import chunk_mesa_net

"""
MAKO - Local MLP Updates with Online Pseudo-Gradients

Instead of backproping through nonlinear operations, L(v_pred, v), we self-supervise each layer
W_in maps k -> z_in
W_out maps z_out -> v
Then, m(q) = W_out*sigma(W_in q)

This means W_in and W_out can be learned with linear attention rules from FLA.

Here, z_in and z_out are computed just like keys & values. 
Ideally these should approximate the backprop, but maybe its not essential. 
- We can do a full backward of k -> v once every chunk

"""

class Mako(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        expand_v: int = 1,
        num_heads: int = 4,
        dim_inner = 4,
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

        self.use_rope = use_rope

        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_zin = nn.Linear(self.dim, self.dim_inner*self.num_heads, bias=False)
        self.proj_zout = nn.Linear(self.dim, self.dim_inner*self.num_heads, bias=False)

        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim)
        self.k_conv1d = ShortConvolution(self.key_dim)
        self.v_conv1d = ShortConvolution(self.value_dim)
        self.zin_conv1d = ShortConvolution(self.dim_inner*self.num_heads)
        self.zout_conv1d = ShortConvolution(self.dim_inner*self.num_heads)


        self.rule = 'mesa_rule' #alternatively, 'mesa_rule'
        self.gate_proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.head_v_dim)
        

        
        if self.rule == 'gated_delta_rule':
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_in_log = nn.Parameter(torch.log(A))
            self.A_in_log._no_weight_decay = True
            self.A_out_log = nn.Parameter(torch.log(A))
            self.A_out_log._no_weight_decay = True
            # hard coded for now
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min),
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_in_bias = nn.Parameter(inv_dt)
            self.dt_in_bias._no_weight_decay = True
            self.dt_out_bias = nn.Parameter(inv_dt)
            self.dt_out_bias._no_weight_decay = True

        elif self.rule == 'gated_delta_rule':
            self.lambda_lower_bound = 0.25
            lambda_initial_value = 1.0
            init_lamb_value = torch.log(torch.exp(torch.tensor(lambda_initial_value - self.lambda_lower_bound)) - 1.0)
            init_lamb_params = torch.empty(self.key_dim, dtype=torch.float32).fill_(init_lamb_value)

            self.lambda_params = nn.Parameter(init_lamb_params)
            self.lambda_params._no_weight_decay = True
            

        self.b_in_proj = nn.Linear(self.dim, self.num_heads, bias=False)
        self.b_out_proj = nn.Linear(self.dim, self.num_heads, bias=False)
        self.weight_decay_in_proj = nn.Linear(self.dim, self.num_heads, bias=False)
        self.weight_decay_out_proj = nn.Linear(self.dim, self.num_heads, bias=False)



    def forward(self,
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        dtype = torch.float32 #torch.bfloat16
        b, l = hidden_states.size()[:2]
        q = self.q_conv1d(self.proj_q(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(dtype).contiguous()
        k = self.k_conv1d(self.proj_k(hidden_states)).view(b, l, self.num_heads, self.head_qk_dim).to(dtype).contiguous()
        v = self.v_conv1d(self.proj_v(hidden_states)).view(b, l, self.num_heads, self.head_v_dim).to(dtype).contiguous()
        
        z_in = self.zin_conv1d(self.proj_zin(hidden_states)).view(b, l, self.num_heads, self.dim_inner).to(dtype).contiguous()
        z_out = self.zout_conv1d(self.proj_zout(hidden_states)).view(b, l, self.num_heads, self.dim_inner).to(dtype).contiguous()

        if self.use_rope: #note: used to cast q,k,v in bfloat16 AFTER applying rope.
            pos = torch.arange(l, device=hidden_states.device)
            q, k = apply_rope(q, k, positions=pos, base=10000.0)

        beta_in = self.b_in_proj(hidden_states).sigmoid().to(dtype).contiguous()
        beta_out = self.b_out_proj(hidden_states).sigmoid().to(dtype).contiguous()
        decay_in = self.weight_decay_in_proj(hidden_states).to(dtype).contiguous()
        decay_out = self.weight_decay_out_proj(hidden_states).to(dtype).contiguous()

        if self.rule == 'gated_delta_rule':
            y, _ = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=z_in,
                g=decay_in,
                beta=beta_in,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                A_log=self.A_in_log,
                dt_bias=self.dt_in_bias,
            )

            """ Note: You can add any nonlinear activation here if you want """
            #y = y.softmax(dim=-1) #Alternatively, use different activation for the online-trained MLP.
            y = F.relu(y)

            o, _ = chunk_gated_delta_rule(
                q=y,
                k=z_out,
                v=v,
                g=decay_out,
                beta=beta_out,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                A_log=self.A_out_log,
                dt_bias=self.dt_out_bias,
            )

        elif self.rule == 'mesa_rule':
            lamb = F.softplus(self.lambda_params.float()) + self.lambda_lower_bound
            lamb = lamb.reshape(self.num_heads, -1)
            y, h_kk, h_kv = chunk_mesa_net(
                q=q,
                k=k,
                v=z_in,
                g=F.logsigmoid(decay_in),
                beta=beta_in,
                lamb=lamb,
                max_CG_iteration=30,
                use_qk_l2norm_in_kernel=True,
            )
            y = F.relu(y)
            o, h_kk, h_kv = chunk_mesa_net(
                q=y,
                k=z_out,
                v=v,
                g=F.logsigmoid(decay_out),
                beta=beta_out,
                lamb=lamb,
                max_CG_iteration=30,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            #raise not implemented error
            raise

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

