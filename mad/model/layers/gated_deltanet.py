import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fla.layers.gated_deltanet import GatedDeltaNet



class GatedDeltaNetAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        expand_v: int = 2,
        num_heads: int = 4,
        **kwargs
    ):
        super().__init__()
        self.model = GatedDeltaNet(
            hidden_size=dim,
            expand_v=expand_v,
            head_dim=dim//num_heads,
            num_heads=num_heads,
            num_v_heads=None,
            mode='chunk',
            use_gate=True,
            use_short_conv=True,
            allow_neg_eigval=False,
            conv_size=4,
            conv_bias=False,
            layer_idx=None,
            norm_eps=1e-5,
        )

    def forward(self,
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        out, _, _ = self.model(hidden_states)
        return out

        