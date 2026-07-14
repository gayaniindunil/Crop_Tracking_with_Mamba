"""Selective state-space (S6 / Mamba) primitive.

`selective_scan_ref` is a readable O(L) reference of the S6 recurrence. In
production replace it with the fused CUDA kernel `selective_scan_fn` from the
`mamba_ssm` package (import guarded below) for ~10-50x speedup.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:                                              # optional fast path
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _HAS_FAST = True
except Exception:                                 # pragma: no cover
    _HAS_FAST = False


def selective_scan_ref(u, delta, A, B, C, D=None):
    """S6 recurrence. u,delta:(b,d,L)  A:(d,n)  B,C:(b,n,L)  D:(d,)."""
    b, d, L = u.shape
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaBu = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    x = u.new_zeros((b, d, A.shape[-1]))
    ys = []
    for t in range(L):
        x = deltaA[:, :, t] * x + deltaBu[:, :, t]        # h_t = A_bar h_{t-1}+B_bar u_t
        ys.append(torch.einsum('bdn,bn->bd', x, C[:, :, t]))
    y = torch.stack(ys, dim=-1)
    return y + u * D.view(1, -1, 1) if D is not None else y


class S6(nn.Module):
    """Selective scan over an arbitrary token axis, with optional step scaling.

    `delta_scale` (b,L,1) modulates the input-dependent step: values > 1 make the
    state refresh faster (used under low alignment confidence), values < 1 make it
    persist (the Occlusion-Persistent State behaviour when the net emits delta->0).
    """
    def __init__(self, d_inner, d_state=16, dt_rank=None, use_fast=True):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        self.dt_rank = dt_rank or max(1, d_inner // 16)
        self.use_fast = use_fast and _HAS_FAST
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A.contiguous()))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, u, delta_scale=None):           # u:(b,d_inner,L)
        A = -torch.exp(self.A_log)
        proj = self.x_proj(rearrange(u, 'b d l -> b l d'))
        dt, B, C = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))          # (b,L,d_inner)
        if delta_scale is not None:
            delta = delta * delta_scale
        delta = rearrange(delta, 'b l d -> b d l')
        B = rearrange(B, 'b l n -> b n l')
        C = rearrange(C, 'b l n -> b n l')
        if self.use_fast and delta_scale is None:     # fast kernel path
            return selective_scan_fn(u, delta, A, B, C, self.D.float())
        return selective_scan_ref(u, delta, A, B, C, self.D)
