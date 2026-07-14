"""Decoupled detection / ordinal-phenology / instance-embedding heads + FPN neck."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PAFPN(nn.Module):
    """Minimal path-aggregation FPN over the 4-level pyramid -> single fused level."""
    def __init__(self, dims=(96, 192, 384, 768), out=256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(d, out, 1) for d in dims])
        self.smooth = nn.ModuleList([nn.Conv2d(out, out, 3, 1, 1) for _ in dims])

    def forward(self, feats):                      # list of (b,T,C_i,H_i,W_i)
        b, T = feats[0].shape[:2]
        xs = [self.lat[i](f.flatten(0, 1)) for i, f in enumerate(feats)]
        for i in range(len(xs) - 1, 0, -1):        # top-down
            xs[i - 1] = xs[i - 1] + F.interpolate(xs[i], size=xs[i - 1].shape[-2:], mode='nearest')
        xs = [self.smooth[i](x) for i, x in enumerate(xs)]
        out = xs[1]                                # P3-equivalent (stride 8)
        return rearrange(out, '(b t) c h w -> b t c h w', b=b, t=T)


class DetectionHead(nn.Module):
    """Anchor-free per-frame detector (objectness + DFL box)."""
    def __init__(self, dim, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.stem = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.SiLU())
        self.obj = nn.Conv2d(dim, 1, 1)
        self.box = nn.Conv2d(dim, 4 * (reg_max + 1), 1)

    def forward(self, f):
        b, T = f.shape[:2]
        x = self.stem(f.flatten(0, 1))
        r = lambda z: rearrange(z, '(b t) c h w -> b t c h w', b=b, t=T)
        return {'obj': r(self.obj(x)), 'box': r(self.box(x))}


class OrdinalPhenologyHead(nn.Module):
    """Growth stage (ordinal CORAL) + continuous ripeness + days-to-ripe."""
    def __init__(self, dim, n_stages=6):
        super().__init__()
        self.n_stages = n_stages
        self.stem = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.SiLU())
        self.ordi = nn.Conv2d(dim, n_stages - 1, 1)
        self.ripe = nn.Conv2d(dim, 1, 1)
        self.dtr = nn.Conv2d(dim, 1, 1)

    def forward(self, f):
        b, T = f.shape[:2]
        x = self.stem(f.flatten(0, 1))
        r = lambda z: rearrange(z, '(b t) c h w -> b t c h w', b=b, t=T)
        return {'stage_logits': r(self.ordi(x)),
                'ripeness': torch.sigmoid(r(self.ripe(x))),
                'days_to_ripe': F.softplus(r(self.dtr(x)))}


class InstanceEmbedHead(nn.Module):
    """Decoupled L2-normalised ID descriptor."""
    def __init__(self, dim, d_embed=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.SiLU())
        self.emb = nn.Conv2d(dim, d_embed, 1)

    def forward(self, f):
        b, T = f.shape[:2]
        e = F.normalize(self.emb(self.stem(f.flatten(0, 1))), dim=1)
        return rearrange(e, '(b t) c h w -> b t c h w', b=b, t=T)
