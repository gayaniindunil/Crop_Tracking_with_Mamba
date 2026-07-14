"""Spatial cross-scan, deformable alignment, and the Dual-Arrow temporal scan."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .ssm import S6


class SpatialCrossScan(nn.Module):
    """VMamba-style 4-directional SS2D per frame -> omnidirectional field."""
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.scan = S6(dim, d_state)
        self.merge = nn.Linear(dim, dim)

    @staticmethod
    def _dirs(z):                                  # z:(b,d,H,W)
        b, d, H, W = z.shape
        h = z.reshape(b, d, H * W)
        hv = z.permute(0, 1, 3, 2).reshape(b, d, H * W)
        return [h, torch.flip(h, [-1]), hv, torch.flip(hv, [-1])]

    def forward(self, z):
        b, d, H, W = z.shape
        o = [self.scan(s) for s in self._dirs(z)]
        o0 = o[0]
        o1 = torch.flip(o[1], [-1])
        o2 = o[2].reshape(b, d, W, H).permute(0, 1, 3, 2).reshape(b, d, H * W)
        o3 = torch.flip(o[3], [-1]).reshape(b, d, W, H).permute(0, 1, 3, 2).reshape(b, d, H * W)
        y = (o0 + o1 + o2 + o3) * 0.25
        y = self.merge(rearrange(y, 'b d (h w) -> b h w d', h=H))
        return rearrange(y, 'b h w d -> b d h w')


class DeformableAlign(nn.Module):
    """Warp each frame toward a reference frame with a learned dense offset field,
    and emit a per-location alignment confidence. Absorbs the depth-dependent
    parallax residual left by an occasional 2-3 cm camera translation. Lightweight
    hidden channels keep the parameter cost modest; run at coarse stages."""
    def __init__(self, dim, max_shift=6):
        super().__init__()
        self.max_shift = max_shift
        ho, hc = max(16, dim // 4), max(8, dim // 8)
        self.offset = nn.Sequential(nn.Conv2d(2 * dim, ho, 3, 1, 1), nn.SiLU(),
                                    nn.Conv2d(ho, 2, 3, 1, 1))
        self.conf = nn.Sequential(nn.Conv2d(2 * dim, hc, 3, 1, 1), nn.SiLU(),
                                  nn.Conv2d(hc, 1, 1), nn.Sigmoid())

    def forward(self, z, ref_idx=None):            # z:(b,T,d,H,W)
        b, T, d, H, W = z.shape
        ref = z[:, T // 2 if ref_idx is None else ref_idx]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=z.device),
                                torch.linspace(-1, 1, W, device=z.device), indexing='ij')
        base = torch.stack([xs, ys], -1)[None]
        rad = 2.0 * self.max_shift / max(H, W)
        aligned, confs = [], []
        for t in range(T):
            pair = torch.cat([z[:, t], ref], dim=1)
            grid = base + torch.tanh(self.offset(pair)).permute(0, 2, 3, 1) * rad
            aligned.append(F.grid_sample(z[:, t], grid, align_corners=True))
            confs.append(self.conf(pair))
        return torch.stack(aligned, 1), torch.stack(confs, 1)


class DualArrowTemporalScan(nn.Module):
    """Forward (causal phenology) + backward (retrospective association) scans over
    per-location, drift-compensated day sequences; fused by a learned gate. The
    alignment confidence modulates the SSM step (gamma) so bumped days refresh."""
    def __init__(self, dim, d_state=16, max_shift=6, gamma=2.0, align=True):
        super().__init__()
        self.align = DeformableAlign(dim, max_shift) if align else None
        self.gamma = gamma
        self.fwd = S6(dim, d_state)
        self.bwd = S6(dim, d_state)
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, z):                          # z:(b,T,d,H,W)
        b, T, d, H, W = z.shape
        ds = None
        if self.align is not None:
            z, conf = self.align(z)
            ds = rearrange(1.0 + self.gamma * (1.0 - conf), 'b t 1 h w -> (b h w) t 1')
        seq = rearrange(z, 'b t d h w -> (b h w) d t')
        yf = self.fwd(seq, ds)
        yb = torch.flip(self.bwd(torch.flip(seq, [-1]),
                                 torch.flip(ds, [1]) if ds is not None else None), [-1])
        cat = rearrange(torch.cat([yf, yb], dim=1), 'n c t -> n t c')
        y = self.proj(torch.sigmoid(self.gate(cat)) * cat)
        return rearrange(y, '(b h w) t d -> b t d h w', b=b, h=H, w=W), (yf, yb)
