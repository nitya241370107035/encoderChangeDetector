"""
backend/services/temporal_head.py
=================================
PyTorch Neural Network Head for Temporal Semantic Change Classification.
Takes bi-temporal Prithvi-EO-2.0-300M latent displacement vectors and physical
spectral deltas, and outputs calibrated multi-class transition probabilities.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Exact 7 Real-World Problem Statement Transition Classes (SIH-2026-PS26227)
SEMANTIC_TRANSITION_CLASSES = [
    "Water Reclamation / Coastal Infrastructure",
    "Urban / Infrastructure Expansion",
    "Land Inundation / Flooding / Submergence",
    "Afforestation / Revegetation / Greening",
    "Deforestation / Demolition / Land Clearing",
    "Cryospheric Dynamics (Snow & Ice Shift)",
    "Wetland & Ecological Succession"
]


def expand_spectral_features(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Expands 3 raw spectral deltas [d_ndvi, d_ndbi, d_ndwi] into 11 physical non-linear
    interaction features:
      [s (3), |d_ndvi|, |d_ndbi|, |d_ndwi|,
       d_ndvi - d_ndbi, d_ndvi - d_ndwi, d_ndwi - d_ndbi,
       d_ndvi * d_ndbi, d_ndvi * d_ndwi]
    Turns a 2051-D vector into a 2059-D vector.
    """
    if isinstance(x, torch.Tensor):
        if x.shape[-1] != 2051:
            return x
        latent = x[:, :2048]
        s = x[:, 2048:]
        d_ndvi = s[:, 0:1]
        d_ndbi = s[:, 1:2]
        d_ndwi = s[:, 2:3]
        expanded = torch.cat([
            s,
            torch.abs(d_ndvi), torch.abs(d_ndbi), torch.abs(d_ndwi),
            d_ndvi - d_ndbi,
            d_ndvi - d_ndwi,
            d_ndwi - d_ndbi,
            d_ndvi * d_ndbi,
            d_ndvi * d_ndwi
        ], dim=-1)
        return torch.cat([latent, expanded], dim=-1)
    elif isinstance(x, np.ndarray):
        if x.shape[-1] != 2051:
            return x
        latent = x[:, :2048]
        s = x[:, 2048:]
        d_ndvi = s[:, 0:1]
        d_ndbi = s[:, 1:2]
        d_ndwi = s[:, 2:3]
        expanded = np.concatenate([
            s,
            np.abs(d_ndvi), np.abs(d_ndbi), np.abs(d_ndwi),
            d_ndvi - d_ndbi,
            d_ndvi - d_ndwi,
            d_ndwi - d_ndbi,
            d_ndvi * d_ndbi,
            d_ndvi * d_ndwi
        ], axis=-1)
        return np.concatenate([latent, expanded], axis=-1)
    else:
        return x


class TemporalChangeHead(nn.Module):
    """
    SOTA Multi-Branch Gated Residual Head for Bi-Temporal Semantic Change Detection.
    Features:
    - Pathway A: Latent Foundation Displacement [delta_e || e_before] (2048 dims) -> 512 dims LayerNorm
    - Pathway B: Non-Linear Biophysical Spectral Physics (11 dims) -> 128 dims LayerNorm
    - Multiplicative Sigmoid Biophysical Gate (modulates 512-D latent space by physical index shifts)
    - 2x Residual Blocks with LayerNorm, GELU, and Dropout
    - Multi-class Calibrated Semantic Classifier (7 PS classes)
    """

    def __init__(self, in_dim: int = 2059, num_classes: int = 7, dropout: float = 0.25):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.spec_dim = in_dim - 2048

        # Latent Foundation Embedding Pathway (2048 dims)
        self.latent_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Biophysical Spectral Physics Pathway (11 dims)
        self.spec_proj = nn.Sequential(
            nn.Linear(self.spec_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6)
        )

        # Biophysical Cross-Attention Multiplicative Gate
        self.gate = nn.Sequential(
            nn.Linear(128, 512),
            nn.Sigmoid()
        )

        # Cross-Pathway Residual Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2x Deep Residual Blocks with LayerNorm & GELU
        self.res1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

        self.res2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

        # Semantic Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (Batch, in_dim). If in_dim == 2051, auto-expands to 2059.
        """
        if x.shape[-1] == 2051 and self.in_dim == 2059:
            x = expand_spectral_features(x)

        feat_latent = x[:, :2048]
        feat_spec = x[:, 2048:]

        h_lat = self.latent_proj(feat_latent)
        h_spec = self.spec_proj(feat_spec)

        g = self.gate(h_spec)
        h_gated = h_lat * (1.0 + g)

        h = torch.cat([h_gated, h_spec], dim=-1)
        h = self.fusion(h)
        h = F.gelu(h + self.res1(h))
        h = F.gelu(h + self.res2(h))

        return self.classifier(h)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized probability distribution across semantic transition classes."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def load_trained_head(
    checkpoint_path: str = "models/change_head/temporal_change_head.pth",
    device: Optional[str] = None
) -> Tuple[Optional[TemporalChangeHead], List[str]]:
    """Loads trained checkpoint and class vocabulary if available."""
    default_classes = SEMANTIC_TRANSITION_CLASSES

    if not os.path.exists(checkpoint_path):
        return None, default_classes

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    in_dim = ckpt.get("in_dim", 2059)
    num_classes = ckpt.get("num_classes", len(default_classes))
    classes = ckpt.get("classes", default_classes)

    model = TemporalChangeHead(in_dim=in_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    return model, classes

