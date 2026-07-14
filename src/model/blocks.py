"""ST-SSM block, patch stem, downsample, and the hierarchical backbone."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from einops import rearrange
from .scans import SpatialCrossScan, DualArrowTemporalScan


class STSSMBlock(nn.Module):
    def __init__(self, dim, d_state=16, mlp_ratio=2.0, align=True, max_shift=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.spatial = SpatialCrossScan(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.temporal = DualArrowTemporalScan(dim, d_state, max_shift=max_shift, align=align)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    @staticmethod
    def _ln(norm, z):
        return rearrange(norm(rearrange(z, 'b t d h w -> b t h w d')), 'b t h w d -> b t d h w')

    def forward(self, z):                          # z:(b,T,d,H,W)
        b, T, d, H, W = z.shape
        s = self._ln(self.norm1, z)
        s = self.spatial(s.reshape(b * T, d, H, W)).view(b, T, d, H, W)
        z = z + s
        t_out, streams = self.temporal(self._ln(self.norm2, z))
        z = z + t_out
        m = self.mlp(rearrange(self._ln(self.norm3, z), 'b t d h w -> b t h w d'))
        return z + rearrange(m, 'b t h w d -> b t d h w'), streams


class PatchStem(nn.Module):
    def __init__(self, in_ch=3, dim=96):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, dim // 2, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(dim // 2, dim, 3, 2, 1))

    def forward(self, x):                          # x:(b,T,3,H,W)
        b, T = x.shape[:2]
        return rearrange(self.proj(x.flatten(0, 1)), '(b t) d h w -> b t d h w', b=b, t=T)


class Downsample(nn.Module):
    def __init__(self, di, do):
        super().__init__()
        self.conv = nn.Conv2d(di, do, 3, 2, 1)

    def forward(self, z):
        b, T = z.shape[:2]
        return rearrange(self.conv(z.flatten(0, 1)), '(b t) d h w -> b t d h w', b=b, t=T)


class PhenoMambaBackbone(nn.Module):
    """4-stage hierarchy. `align_stages` selects which stages use DeformableAlign
    (default: only the two coarse stages, where camera-motion residual is a few
    tokens and the cost is amortised)."""
    def __init__(self, dims=(96, 192, 384, 768), depths=(2, 2, 6, 2), d_state=16,
                 align_stages=(2, 3), max_shift=6):
        super().__init__()
        self.stem = PatchStem(3, dims[0])
        self.stages, self.downs = nn.ModuleList(), nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            use_align = i in align_stages
            self.stages.append(nn.ModuleList(
                [STSSMBlock(dim, d_state, align=use_align, max_shift=max_shift) for _ in range(depth)]))
            if i < len(dims) - 1:
                self.downs.append(Downsample(dim, dims[i + 1]))

    def forward(self, x) -> List[torch.Tensor]:
        z = self.stem(x)
        feats = []
        for i, stage in enumerate(self.stages):
            for blk in stage:
                z, _ = blk(z)
            feats.append(z)
            if i < len(self.downs):
                z = self.downs[i](z)
        return feats
