import torch

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
