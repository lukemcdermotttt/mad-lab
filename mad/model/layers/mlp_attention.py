# adpated from:
# https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/linear_attn.py
# https://github.com/HazyResearch/based/blob/main/based/models/mixers/linear_attention.py

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
#from einops import rearrange
from flash_attn import flash_attn_func, flash_attn_with_kvcache

#from fla.ops.delta_rule import chunk_delta_rule
import math 

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

from mad.model.layers.ops.rope import apply_rope

class MLPAttention(nn.Module):
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
        dim_inner = 4,
        chunk_size: int = 16,
        use_weight_decay: bool = True,
        num_backprop_steps: int = 1,
        base_lr: float = 1e-3,
        base_weight_decay: float = 0.9,
        use_self_distillation: bool = False,
        self_distillation_type: str = 'standard',
        use_trainable_init: bool = True,
        fusion: str = 'dynamic',
        use_short_conv: bool = False,
        short_conv_use_norm: bool = False,
        use_lola: bool = False,
        sparse_cache_size: int = 16,
        use_rope: bool = False,
        use_momentum: bool = False,
        use_muon: bool = False,
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
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.proj_beta = nn.Linear(self.dim, self.num_heads*2, bias=False)
        self.proj_lr = nn.Linear(self.dim, self.num_heads*2, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)

        self.use_short_conv = use_short_conv
        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=4,
                bias=False,
                activation='silu',
                use_norm=short_conv_use_norm
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=4,
                bias=False,
                activation='silu',
                use_norm=short_conv_use_norm,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=4,
                bias=False,
                activation='silu',
                use_norm=short_conv_use_norm,
            )

        self.norm_q = norm_q
        self.norm_k = norm_k
        
        self.dim_inner = dim_inner
        self.fusion = fusion
        self.chunk_size = chunk_size
        self.base_lr = base_lr
        self.base_weight_decay = base_weight_decay
        self.num_backprop_steps = num_backprop_steps
        self.use_weight_decay = use_weight_decay
        self.use_lola = use_lola
        self.sparse_cache_size = sparse_cache_size
        self.use_self_distillation = use_self_distillation
        self.self_distillation_type = self_distillation_type
        self.use_rope = use_rope
        self.use_trainable_init = use_trainable_init
        if self.use_trainable_init:
            self.W_in_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_qk_dim)) / math.sqrt(self.head_qk_dim))
            self.W_out_init = nn.Parameter(torch.randn((1,self.dim_inner,self.num_heads,self.head_v_dim)) / math.sqrt(self.dim_inner))
        self.use_momentum = use_momentum
        if self.use_momentum:
            self.proj_momen = nn.Sequential(nn.Linear(self.dim, self.num_heads), nn.Sigmoid())
        self.use_muon = use_muon
        

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
        b, l = hidden_states.size()[:2]
        beta, lr = self.proj_beta(hidden_states), self.proj_lr(hidden_states)

        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.proj_q(hidden_states))
            k, _ = self.k_conv1d(x=self.proj_k(hidden_states))
            v, _ = self.v_conv1d(x=self.proj_v(hidden_states))
        else:
            q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)


        q = q.view(b, l, self.num_heads, self.head_qk_dim)
        k = k.view(b, l, self.num_heads, self.head_qk_dim)
        v = v.view(b, l, self.num_heads, self.head_v_dim)

        # global RoPE over the full sequence
        if self.use_rope:
            pos = torch.arange(l, device=hidden_states.device)  # (l,)
            q, k = apply_rope(q, k, positions=pos, base=10000.0)

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr).to(q.dtype)
        beta = torch.sigmoid(torch.nn.functional.softplus(beta.float() + self.base_weight_decay))
        beta, lr = beta.view(b, l, self.num_heads, 2), lr.view(b, l, self.num_heads, 2)
        beta_in, beta_out = beta[:,:,:,0], beta[:,:,:,1]
        lr_in, lr_out = lr[:,:,:,0], lr[:,:,:,1]
        
        q, k = self.feature_map_q(q), self.feature_map_k(k)
        if self.norm_q:
            q = q / (q.sum(-1, keepdim=True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, keepdim=True) + 1e-4)


        o = torch.zeros_like(v)

        if self.use_trainable_init:
            W_in = self.W_in_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
            W_out = self.W_out_init.expand(b,-1,-1,-1).to(q.device).to(torch.bfloat16)
        else:
            W_in = torch.zeros((b,self.dim_inner,self.num_heads,self.head_qk_dim), device=q.device, dtype=torch.bfloat16)
            W_out = torch.zeros((b,self.dim_inner,self.num_heads,self.head_v_dim), device=q.device, dtype=torch.bfloat16)
        if self.use_momentum:
            momen_in = torch.zeros_like(W_in)
            momen_out = torch.zeros_like(W_out)
            momen_coeff = self.proj_momen(hidden_states) #(b,l,h)


        lola_keys = torch.zeros((b,0,self.num_heads,self.head_qk_dim), device=q.device, dtype=torch.bfloat16)
        lola_values = torch.zeros((b,0,self.num_heads,self.head_v_dim), device=q.device, dtype=torch.bfloat16)
        
        for i in range(0, l, self.chunk_size):
            start = i #max(i-self.chunk_size,0)
            end   = min(i+self.chunk_size, l+1)
            #Forward Pass of the Query
            q_chunk = torch.cat([W_in, lola_keys, q[:,start:end, :, :]], dim=1).to(torch.bfloat16) #(We will mask out the first self.dim_inner + lola_keys positions after)
            k_chunk = torch.cat([W_in, lola_keys, k[:, start:end, :, :]], dim=1).to(torch.bfloat16)
            v_chunk = torch.cat([W_out, lola_values, v[:, start:end, :, :]], dim=1).to(torch.bfloat16)
            
            if self.fusion == 'dynamic': #Our Approach
                o[:,i:i+self.chunk_size] = flash_attn_func(q_chunk, k_chunk, v_chunk, causal=True)[:,self.dim_inner + lola_keys.size(1):] 
            elif self.fusion == 'static': #LaCT Approach
                assert self.use_lola == False, "LoLA not yet implemented for static"
                o[:,i:i+self.chunk_size] = flash_attn_func(q[:, start:end].to(torch.bfloat16), k[:,start:end].to(torch.bfloat16), v[:,start:end].to(torch.bfloat16), causal=True)[:,-self.chunk_size:].contiguous()
                o[:,i:i+self.chunk_size] += l2_norm(flash_attn_func(q[:,i:end].to(torch.bfloat16), W_in.to(torch.bfloat16), W_out.to(torch.bfloat16), causal=False))
            else:
                raise NotImplementedError

            #Backward Pass for Long Term Memory
            for j in range(self.num_backprop_steps):
                inputs = k_chunk[:,self.dim_inner: ] #Current chunk of keys, potentially including lola_keys
                targets = v_chunk[:,self.dim_inner: ] #Current chunk of values, potentially including lola_values
                #Grad_in, Grad_out, losses = mlp_backprop_flash(inputs, targets, W_in, W_out)
                Grad_in, Grad_out, losses = mlp_backprop(inputs, targets, W_in, W_out, compute_losses=self.use_lola) #if self.use_lola is false, losses = None
                #Grad_in, Grad_out, losses = huber_mlp_backprop(inputs, targets, W_in, W_out, compute_losses=self.use_lola, robust='tukey') #use tukey loss!

                if self.use_lola:
                    sorted_idx = losses.argsort(dim=1, descending=True)[:,:self.sparse_cache_size] #Get the indices of the top-k losses. shape = (b, chunk_size)
                    lola_keys = torch.gather(inputs, dim=1, index=sorted_idx[:,:,:,None].expand(-1,-1,self.num_heads,self.head_qk_dim)) #Get the top-k inputs. shape = (b, sparse_cache_size, h, head_qk_dim)
                    lola_values = torch.gather(targets, dim=1, index=sorted_idx[:,:,:,None].expand(-1,-1,self.num_heads,self.head_v_dim)) #Get the top-k targets. shape = (b, sparse_cache_size, h, head_v_dim)
                else:
                    assert lola_keys.size(1) == 0 and lola_values.size(1) == 0, "lola_keys and lola_values should be empty when use_lola is False"

                if self.use_self_distillation:
                    if self.self_distillation_type == 'huber':
                        Grad_in_qy, Grad_out_qy, _ = huber_mlp_backprop(q[:, i:i+self.chunk_size, :, :], o[:, i:i+self.chunk_size, :, :], W_in, W_out)
                    elif self.self_distillation_type == 'tukey':
                        Grad_in_qy, Grad_out_qy, _ = huber_mlp_backprop(q[:, i:i+self.chunk_size, :, :], o[:, i:i+self.chunk_size, :, :], W_in, W_out, robust='tukey')
                    elif self.self_distillation_type == 'rl':
                        with torch.no_grad():
                            y_target = flash_attn_func(q_chunk, k_chunk, v_chunk, causal=False)[:,(self.dim_inner+lola_keys.size(1)):] 
                        #y_target = flash_attn_func(q_chunk, k_chunk, v_chunk, causal=False)[:,(self.dim_inner+lola_keys.size(1)):] 
                        Grad_in_qy, Grad_out_qy, _ = mlp_backprop(q[:, i:i+self.chunk_size, :, :], y_target, W_in, W_out)

                    else:
                        Grad_in_qy, Grad_out_qy, _ = mlp_backprop(q[:, i:i+self.chunk_size, :, :], o[:, i:i+self.chunk_size, :, :], W_in, W_out)
                    Grad_in += Grad_in_qy #* (i/l)
                    Grad_out += Grad_out_qy#* (i/l)

                if self.use_momentum:
                    m_chunk = momen_coeff[:,i:i+self.chunk_size].mean(dim=1,keepdim=True)[:,:,:,None]
                    Grad_in = Grad_in + m_chunk * momen_in 
                    Grad_out = Grad_out + m_chunk * momen_out
                    momen_in = Grad_in
                    momen_out = Grad_out

                    if self.use_muon:
                        Grad_in = zeropower_via_newtonschulz5(Grad_in)
                        Grad_out = zeropower_via_newtonschulz5(Grad_out)

                if self.use_weight_decay:
                    assert False, "Forgot to take mean over lr / beta in the chunk for the Gradient step"
                    W_in = (beta_in[:,i:i+1,:,None] * W_in) - lr_in[:,i:i+1,:,None] * Grad_in
                    W_out = (beta_out[:,i:i+1,:,None] * W_out) - lr_out[:,i:i+1,:,None] * Grad_out
                else:
                    assert False, "Forgot to take mean over lr / beta in the chunk for the Gradient step"
                    W_in = W_in - lr_in[:,i:i+1,:,None] * Grad_in
                    W_out = W_out - lr_out[:,i:i+1,:,None] * Grad_out

        o = o.contiguous().view(b, l, -1)
        o = self.out_proj(o)
        return o


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


def mlp_backprop(x, y, W_in, W_out, compute_losses=False):
    z = torch.einsum('bnhd,bDhd->bnhD', x, W_in)
    p = torch.softmax(z, dim=1)
    Grad_out = torch.einsum('bnhD,bnhd->bDhd', -p, y)

    a = torch.einsum('bDhd,bnhd->bnhD', W_out, y)
    s = (p*a).sum(dim=-1, keepdim=True)
    g_z = -(p*a - p*s)
    Grad_in = torch.einsum('bnhd,bnhD->bDhd', x, g_z)

    if compute_losses:
        losses = -(p*a).sum(dim=3) #Loss per KV pair per head. shape = (b, l, h)
        return Grad_in, Grad_out, losses
    return Grad_in, Grad_out, None

def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)



def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    and further used from https://github.com/a1600012888/LaCT/blob/main/lact_llm/lact_model/ttt_operation.py#L31
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3

    Updating to 

    [b, d, h, d']
    """
    raise "Currently has wrong dims, takes in torch.Size([128, 16, 4, 32]) and outputs torch.Size([128, 32, 4, 16]) instead. Turn muon off."
    assert len(G.shape) == 4
    X = G.bfloat16()
    if G.size(1) > G.size(3):
        X = X.transpose(1, 3)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 3), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        
        #A = X @ X.transpose(1, 2) #(b, d, d') (b, d', d) -> (b, d, d)
        A = torch.einsum('bXhd,bYhd->bXhY', X, X)
        #B = (b * A + c * A @ A )  #(b,d,d)+(b,d,d)(b,d,d)
        B = (b*A+c*torch.einsum('bXhd,bdhY->bXhY', A, A))
        #X = a * X + B @ X #(b,d,d) (b, d, d') -> (b,d,d')
        X = a*X + torch.einsum('bXhd,bdhY->bXhY', B, X)

    if G.size(1) > G.size(2):
        X = X.transpose(1, 3)
    return X


    # assert len(G.shape) == 3
    # X = G.bfloat16()
    # if G.size(1) > G.size(2):
    #     X = X.transpose(1, 2)
    # # Ensure spectral norm is at most 1
    # X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # # Perform the NS iterations
    # for a, b, c in [
    #     (4.0848, -6.8946, 2.9270),
    #     (3.9505, -6.3029, 2.6377),
    #     (3.7418, -5.5913, 2.3037),
    #     (2.8769, -3.1427, 1.2046),
    #     (2.8366, -3.0525, 1.2012),
    # ]:
    #     A = X @ X.transpose(1, 2)
    #     B = (
    #         b * A + c * A @ A
    #     )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
    #     X = a * X + B @ X

    # if G.size(1) > G.size(2):
    #     X = X.transpose(1, 2)
    # return X


def huber_psi(r: torch.Tensor, delta: float):
    # r: (..., d)
    abs_r = r.abs()
    return torch.where(abs_r <= delta, r, delta * r.sign())

def huber_rho(r: torch.Tensor, delta: float):
    abs_r = r.abs()
    quad = 0.5 * r * r
    lin = delta * (abs_r - 0.5 * delta)
    return torch.where(abs_r <= delta, quad, lin)

def tukey_psi(r: torch.Tensor, c: float):
    # Tukey biweight influence (elementwise)
    # psi(r) = r * (1 - (r/c)^2)^2 for |r|<c else 0
    x = r / c
    mask = (x.abs() < 1.0)
    w = (1.0 - x * x)
    return torch.where(mask, r * (w * w), torch.zeros_like(r))

def tukey_rho(r: torch.Tensor, c: float):
    # Tukey biweight loss (elementwise), up to additive constant
    # rho(r) = (c^2/6) * (1 - (1 - (r/c)^2)^3) for |r|<c else (c^2/6)
    x = r / c
    mask = (x.abs() < 1.0)
    one_minus = (1.0 - x * x)
    inside = (c * c / 6.0) * (1.0 - one_minus * one_minus * one_minus)
    outside = (c * c / 6.0) * torch.ones_like(r)
    return torch.where(mask, inside, outside)

def huber_mlp_backprop(
    x, y_target, W_in, W_out,
    compute_losses: bool = False,
    robust: str = "huber",      # "huber" or "tukey"
    delta: float = 1.0,         # huber delta / tukey c
):
    """
    x:        [b, n, h, d_qk]   (inputs: keys or queries you train on)
    y_target: [b, n, h, d_v]    (targets: values or teacher outputs)
    W_in:     [b, D, h, d_qk]
    W_out:    [b, D, h, d_v]
    """

    # ----- forward of the "MLP attention" memory -----
    z = torch.einsum('bnhd,bDhd->bnhD', x, W_in)          # [b,n,h,D]
    p = torch.softmax(z, dim=1)                           # softmax over n (your design)

    y_hat = torch.einsum('bnhD,bDhd->bnhd', p, W_out)     # [b,n,h,d_v]
    r = (y_hat - y_target)                                # residual

    # ----- robust influence psi(r) = d rho / d y_hat -----
    if robust == "huber":
        g = huber_psi(r, delta)                           # [b,n,h,d_v]
        rho = huber_rho(r, delta)
    elif robust == "tukey":
        g = tukey_psi(r, delta)
        rho = tukey_rho(r, delta)
    else:
        raise ValueError(f"Unknown robust loss: {robust}")

    # ----- backprop using the same structure as your original code -----
    # dL/dW_out: sum_n p_{nD} * g_n
    Grad_out = torch.einsum('bnhD,bnhd->bDhd', p, g)

    # dL/dp_{nD} = <g_n, W_out_D>
    a = torch.einsum('bDhd,bnhd->bnhD', W_out, g)         # [b,n,h,D]
    s = (p * a).sum(dim=-1, keepdim=True)                 # [b,n,h,1]
    g_z = p * (a - s)                                     # [b,n,h,D]  (softmax Jacobian)

    # dL/dW_in: sum_n x_n * g_z_n
    Grad_in = torch.einsum('bnhd,bnhD->bDhd', x, g_z)

    if compute_losses:
        # token/head losses for LoLA selection: [b, n, h]
        losses = rho.sum(dim=-1)                          # sum over d_v
        return Grad_in, Grad_out, losses

    return Grad_in, Grad_out, None


def mlp_backprop_flash(x, y, W_in, W_out):
    """
    x: (b, n, h, d)
    y: (b, n, h, dv)
    W_in: (b, D, h, d)
    W_out: (b, D, h, dv)

    returns:
      Grad_in  = dL/dW_in
      Grad_out = dL/dW_out
    """

    # U = softmax(W_in K^T) V
    U = flash_attn_func(
        W_in.to(torch.bfloat16),
        x.to(torch.bfloat16),
        y.to(torch.bfloat16),
        causal=False
    )

    # Grad_out = -U
    Grad_out = -U

    # Grad_in = - softmax(W_out V^T) K
    Grad_in = -flash_attn_func(
        W_out.to(torch.bfloat16),
        y.to(torch.bfloat16),
        x.to(torch.bfloat16),
        causal=False
    )

    return Grad_in, Grad_out, None