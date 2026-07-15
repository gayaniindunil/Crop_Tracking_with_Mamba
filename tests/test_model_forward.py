import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.phenomamba import PhenoMamba  # noqa: E402


def test_tiny_preset_forward_shapes():
    torch.manual_seed(0)
    model = PhenoMamba.from_preset("tiny").eval()

    b, T, H, W = 1, 2, 64, 64
    x = torch.randn(b, T, 3, H, W)
    with torch.no_grad():
        out = model(x)

    fh, fw = H // 8, W // 8  # PAFPN fuses down to the stride-8 (P3) level
    assert out["obj"].shape == (b, T, 1, fh, fw)
    assert out["embed"].shape == (b, T, model.cfg.d_embed, fh, fw)
    assert torch.isfinite(out["obj"]).all()
