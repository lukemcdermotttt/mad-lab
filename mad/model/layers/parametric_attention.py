import torch
import torch.nn as nn
from flash_attn import flash_attn_func
import torch.nn.functional as F
import math
from mad.model.layers.ops.rope import apply_rope

class ParametricAttention(nn.Module):
    def __init__(self,
        dim : int, # Model dimension
        expand_k : int = 1,
        expand_v : int = 1,
        h : int = 4,  # Number of heads
        d_in : int = 4, # Width of the MLP
        b_chunk : int = 16, # Chunk size
        lr : float = 1e-3,
        **kwargs):
        super().__init__()

        self.d = dim
        self.h = h
        self.d_k = int(dim*expand_k)
        self.d_v = int(dim*expand_v)

        self.d_kh = self.d_k // self.h
        self.d_vh = self.d_v // self.h

        self.d_in = d_in # Width of the MLP
        self.b_chunk = b_chunk # Chunk size
        self.lr = lr # Base learning rate

        self.W_Q  = nn.Linear(in_features=self.d, out_features=self.d_k, bias=False)
        self.W_K  = nn.Linear(in_features=self.d, out_features=self.d_k, bias=False)
        self.W_V  = nn.Linear(in_features=self.d, out_features=self.d_v, bias=False)

        self.W_O = nn.Linear(in_features=self.d_v, out_features=self.d, bias= False) # Projection matrix at the output

        self.Q_conv1d = ShortConvolution(self.d_k)
        self.K_conv1d = ShortConvolution(self.d_k)
        self.V_conv1d = ShortConvolution(self.d_v)

        #self.W_l = nn.Linear(in_features=self.d, out_features=self.h*2, bias = False) # We want to train each layer with different step sizes that can be learned

        self.theta_1_in = nn.Parameter(torch.randn((1, self.d_in, self.h, self.d_kh)) / math.sqrt(self.d_kh))
        self.theta_2_in = nn.Parameter(torch.randn((1, self.d_in, self.h, self.d_vh)) / math.sqrt(self.d_in))

        self.W_G = nn.Linear(in_features=self.d_v, out_features=self.d, bias=False) # Weights learned from input data to gate the attention scores

        self.LayerNorm = nn.LayerNorm(normalized_shape=self.d_vh)

    def forward(self, X: torch.Tensor, *args, **kwargs):

        B, L = X.size()[:2] # Batch size and the length of each training sample

        #Generate Q, K, V

        Q = self.W_Q(X)
        Q = self.Q_conv1d(Q)
        Q = Q.view(B,L,self.h,self.d_kh)
        Q = Q.to(torch.bfloat16).contiguous()


        K = self.W_K(X)
        K = self.K_conv1d(K)
        K = K.view(B, L, self.h, self.d_kh)
        K = K.to(torch.bfloat16).contiguous()

        V = self.W_V(X)
        V = self.V_conv1d(V)
        V = V.reshape(B, L, self.h, self.d_vh)
        V = V.to(torch.bfloat16).contiguous()

        Q, K = apply_rope(Q, K, positions=torch.arange(L, device=X.device), base=10000.0)
        # Learning rates learned from data for each layer of MLP.

        #lr = nn.functional.softplus(self.W_l(X).float() + self.lr)
        #lr = lr.reshape(B, L, self.h, 2)

        O = torch.empty_like(V)

        theta_1 = self.theta_1_in.to(Q.device).to(torch.bfloat16)
        theta_2 = self.theta_2_in.to(Q.device).to(torch.bfloat16)
        chunks = []
        for i in range(0, L, self.b_chunk):
            s, e = i, min(L, i+self.b_chunk)

            theta_1_B = theta_1.expand(B,-1,-1,-1)
            theta_2_B = theta_2.expand(B,-1,-1,-1)

            """Online Forward Pass"""
            chunks.append(flash_attn_func(
                torch.cat([theta_1_B, Q[:, s:e]], dim=1),
                torch.cat([theta_1_B, K[:, s:e]], dim=1),
                torch.cat([theta_2_B, V[:, s:e]], dim=1),
                causal=True)[:, self.d_in:])
            

            """Online Backward Pass"""
            grad_theta_1, grad_theta_2 = compute_gradients(theta_1_B, theta_2_B, K[:,s:e], V[:,s:e])
            theta_1 = theta_1 - self.lr * grad_theta_1
            theta_2 = theta_2 - self.lr * grad_theta_2
        O = torch.cat(chunks, dim=1)       
        O = self.LayerNorm(O)
        O = O.view(B, L, -1)
        out_gate = self.W_G(X)
        O = self.W_O(O * out_gate)
        return O



# ----------------------------------------------------------------------
# Attention via flash-attn (batched, multi-head)
# ----------------------------------------------------------------------
def attention(Q, K, V):
    """
    Attention(Q, K, V) = softmax( Q K^T / sqrt(D_k) ) V,  per (batch, head).

    Q : (B, h, L, d_k)
    K : (B, h, L, d_k)
    V : (B, h, L, d_v)
    -> (B, H, L, d_v)

    flash_attn_func wants (B, L, h, d_k) with d_k == d_v.  We
    zero-pad whichever of {Q, K, V} has the smaller head_dim up to
    D_h = max(D_k, D_v) and pass softmax_scale = 1/sqrt(d_k) so the score
    scaling is unaffected by padding.
    """
    d_k = K.shape[-1]
    d_v = V.shape[-1]
    d_max = max(d_k, d_v)

    bandwidth = 1.0 / (d_k ** 0.5)

    def padding(X, d_max):
        dim = X.shape[-1]
        if dim == d_max:
            return X
        output = torch.zeros(X.shape[0],X.shape[1],X.shape[2],d_max, device=X.device, dtype=X.dtype)
        output[..., :dim] = X
        return output

    # (B, H, S, D) -> (B, S, H, D) for flash-attn.
    q = padding(Q, d_max).permute(0, 2, 1, 3).contiguous()   # (B, L, h, d_max)
    k = padding(K, d_max).permute(0, 2, 1, 3).contiguous()   # (B, L, h, d_max)
    v = padding(V, d_max).permute(0, 2, 1, 3).contiguous()   # (B, L, h, d_max)

    out = flash_attn_func(q, k, v,
                          softmax_scale=bandwidth,
                          causal=False)                     # (B, L, h, d_max)
    return out.permute(0, 2, 1, 3)[..., :d_v]               # (B, h, L, d_v)


def compute_gradients(theta_1, theta_2, K, V):
    """
        Closed-form gradients of
            L = sum_b sum_h sum_i || v_i^{b,h} - (Theta_2^h)^T sigma((Theta_1^h k_i^{b,h})/sqrt(d_k)) ||^2,
        lambda = 0.

        K : (B, L, h, d_k)
        V : (B, L, h, d_v)
        returns (grad_Theta1, grad_Theta2) with shapes (B, H, n, d_k) and (B, H, n, d_v).
        """
    theta_1 = theta_1.permute(0,2,1,3) #(B, h, n, d_k)
    theta_2 = theta_2.permute(0,2,1,3) #(B, h, n, d_v)

    n = theta_1.shape[2]

    K = K.permute(0,2,1,3)
    V = V.permute(0,2,1,3)

    B, h, L, d_k = K.shape

    B, h, L, d_v = V.shape

    inv_sqrt_dk = 1.0 / (d_k ** 0.5)

    # ----- shared attention pieces (each is one fused flash-attn kernel) -----
    # Identity per (batch, head) for Attention(K, Theta_1, I_n).
    I_n = torch.eye(n, device=K.device, dtype=K.dtype)                # (n, n)
    I_BH = I_n.expand(B, h, n, n).contiguous()                        # (B, H, n, n)


    grad_T2 = torch.zeros(1,h,n,d_v,device = K.device, dtype = K.dtype)
    grad_T1 = torch.zeros(1,h,n,d_k, device = K.device, dtype = K.dtype)
    for i in range(0,L,n):
        x, y = i, min(L, i+n)

        K_tilde = torch.zeros(B,h,n,d_k,device=K.device,dtype=K.dtype) 
        K_tilde[:,:,:y-x,:] = K[:,:,x:y,:]
        # att_KI  = attention(K_tilde, theta_1, I_BH)
        att_KI = torch.einsum('bhld,bhnd->bhln',K_tilde,theta_1)[:,:,:y-x,:]                                  # (B, H, l, n)
        att_KT2 = attention(K_tilde, theta_1, theta_2)[:,:,:y-x,:]                                  # (B, H, n, d_v)

        # By Eq. 24, per (batch, head):  sigma(Z) = Attention(K, Theta_1, I)^T
        sigmaZ   = att_KI.transpose(-1, -2)                               # (B, H, n, n)  cols = sigma(z_m)
        sigmaZ_T = att_KI                                                 # (B, H, n, n)  rows = sigma(z_m)^T

        # =====================  Eq. 15:  dL / dTheta_2  =====================
        # Per (batch, head) summand:
        #   -2 * Attention(K, Theta_1, I)^T  V  +  2 * sigma(Z) * Attention(K, Theta_1, Theta_2)
        grad_T2_per_b = 2.0 * (sigmaZ @ (att_KT2 - V[:,:,x:y,:]))                      # (B, H, n, d_v)
        # Theta_2 is shared across B, so total gradient sums over batch.

        grad_T2 += grad_T2_per_b.sum(dim=0,keepdims=True)                                # (1, H, n, d_v)

        # ===================  Eq. 25 - 26:  dL / dTheta_1  ==================
        # Elementwise form. Distributing the j-sum in
        #   R[b,h,i,m] = sum_j (2/sqrt(d_k)) (delta_ij sigma_m[i] - sigma_m[i] sigma_m[j]) Bmat[b,h,j,m]
        # gives
        #   R[b,h,i,m] = (2/sqrt(d_k)) * sigma(z_m)[i] * ( Bmat[b,h,i,m] - <sigma(z_m), Bmat[b,h,:,m]> ).
        # The inner product is the diagonal of  sigmaZ_T @ Bmat, computed as elementwise mul + sum.
        Bmat = theta_2 @ (att_KT2 - V[:,:,x:y,:]).transpose(-1, -2)       # (B, H, n, n)
        PB_diag = (sigmaZ_T * Bmat.transpose(-1, -2)).sum(dim=-1)         # (B, H, n)    -> [b,h,m]
        R = (2.0 * inv_sqrt_dk) * sigmaZ * (Bmat - PB_diag.unsqueeze(-2)) # (B, H, n, n) -> [b,h,i,m]
        grad_T1_per_b = R @ K[:,:,x:y,:]                                  # (B, H, n, d_k)
        grad_T1 += grad_T1_per_b.sum(dim=0,keepdims=True)                                # (1, h, n, d_k)

    if False:
        V_pred = flash_attn_func(K, theta_1, theta_2, causal=False) # (B, L, h, d_v)
        coeff = torch.einsum('blhd,blhd->blh1', V_pred, V_pred - V)
        K = K * coeff
        Z = torch.einsum('blhd,bhnd->blhn', K, theta_1)
        grad_T1 = torch.einsum('blhd,blhn->bhnd', K, Z)

    return grad_T1.permute(0,2,1,3), grad_T2.permute(0,2,1,3)

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