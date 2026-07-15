"""Full PhenoMamba model assembly + config presets."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import torch
import torch.nn as nn
from .blocks import PhenoMambaBackbone
from .heads import PAFPN, DetectionHead, OrdinalPhenologyHead, InstanceEmbedHead
from .association import CrossFrameAssociation


@dataclass
class PhenoCfg:
    dims: Tuple[int, ...] = (96, 192, 384, 768)
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    d_state: int = 16
    neck_out: int = 256
    n_stages: int = 6
    d_embed: int = 128
    align_stages: Tuple[int, ...] = (2, 3)
    max_shift: int = 6


PRESETS = {
    'tiny':  PhenoCfg(dims=(48, 96, 192, 384),  depths=(2, 2, 6, 2), neck_out=192),
    'small': PhenoCfg(dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), neck_out=256),
    'base':  PhenoCfg(dims=(96, 192, 384, 768), depths=(2, 2, 9, 2), neck_out=256),
}


class PhenoMamba(nn.Module):
    def __init__(self, cfg: PhenoCfg = PhenoCfg()):
        super().__init__()
        self.cfg = cfg
        self.backbone = PhenoMambaBackbone(cfg.dims, cfg.depths, cfg.d_state,
                                           cfg.align_stages, cfg.max_shift)
        self.neck = PAFPN(cfg.dims, cfg.neck_out)
        c = cfg.neck_out
        self.det = DetectionHead(c)
        self.phe = OrdinalPhenologyHead(c, cfg.n_stages)
        self.emb = InstanceEmbedHead(c, cfg.d_embed)
        self.assoc = CrossFrameAssociation(cfg.d_embed)

    def forward_features(self, x):                 # x:(b,T,3,H,W) -> fused (b,T,c,H',W')
        return self.neck(self.backbone(x))

    def forward(self, x):
        f = self.forward_features(x)
        out = {'feat': f}
        out.update(self.det(f))
        out.update(self.phe(f))
        out['embed'] = self.emb(f)
        return out                                 # association runs at inference on ROI tokens

    @classmethod
    def from_preset(cls, name='small', **kw):
        cfg = PRESETS[name]
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cls(cfg)


def build_model(name='small', **kw):
    return PhenoMamba.from_preset(name, **kw)
