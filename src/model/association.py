"""Sparse cross-frame association attention on pooled instance tokens + ROI pool."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def roi_pool_embeddings(embed_map, boxes_per_frame, out=1):
    """Pool instance embeddings at detected boxes.
    embed_map: (T, De, H, W) single clip;  boxes_per_frame: list length T of (n_i,4) in
    feature-map coords (x1,y1,x2,y2). Returns tokens (N,De) and frame_idx (N,)."""
    toks, fidx = [], []
    for t, boxes in enumerate(boxes_per_frame):
        if boxes.numel() == 0:
            continue
        idx = torch.zeros((boxes.shape[0], 1), device=boxes.device)
        rois = torch.cat([idx, boxes], dim=1)
        pooled = roi_align(embed_map[t:t + 1], rois, output_size=out, aligned=True)
        toks.append(F.normalize(pooled.flatten(1), dim=-1))
        fidx.append(torch.full((boxes.shape[0],), t, device=boxes.device, dtype=torch.long))
    if not toks:
        return None, None
    return torch.cat(toks, 0), torch.cat(fidx, 0)


class CrossFrameAssociation(nn.Module):
    def __init__(self, d_embed=128, heads=4, layers=2, max_T=64):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_embed, heads, d_embed * 2,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.frame_pos = nn.Parameter(torch.randn(max_T, d_embed) * 0.02)

    def forward(self, tokens, frame_idx):          # (N,De),(N,)
        tok = (tokens + self.frame_pos[frame_idx]).unsqueeze(0)
        ctx = self.enc(tok).squeeze(0)
        ctx = F.normalize(ctx, dim=-1)
        affinity = ctx @ ctx.t()                   # (N,N) cosine affinity
        return ctx, affinity
