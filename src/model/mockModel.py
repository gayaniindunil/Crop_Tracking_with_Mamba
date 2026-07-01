import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests

class MockModel(nn.Module):
    """A lightweight fallback temporal module when `mamba_ssm` is unavailable.

    This provides deterministic, trainable temporal mixing using a single
    bidirectional LSTM followed by a linear projection. The real `mamba_ssm`
    block should be swapped in when available for improved performance.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, num_layers: int = 1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        # Bidirectional LSTM to capture simple temporal context
        self.lstm = nn.LSTM(input_size=dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        out, _ = self.lstm(x)
        out = self.proj(out)
        return out
