"""
backend/services/prithvi_encoder.py
===================================
Prithvi-EO-2.0-300M Vision Foundation Model Encoder Service
===========================================================
Provides a high-performance encoder for multi-spectral Sentinel-2 / HLS satellite
patches using IBM & NASA's Prithvi-EO-2.0-300M foundation model.

- Input: Multi-spectral crops (B02, B03, B04, B05/B08, B06/B11, B07/B12).
- Output: 1024-dimensional L2-normalized semantic latent vector embeddings.
"""

import os
import sys
import json
import logging
import importlib.util
from typing import List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("PrithviEncoder")

MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
CHECKPOINT_FILENAME = "Prithvi_EO_V2_300M.pt"
CONFIG_FILENAME = "config.json"
MAE_FILENAME = "prithvi_mae.py"
EMBEDDING_DIM = 1024

# Default Prithvi-EO-2.0-300M normalisation constants
# Raw DN (0-10000 range)
DATA_MEAN_DN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
DATA_STD_DN = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)

# Fixed reflectance scale [0.0, 1.5] (divided by 10000)
DATA_MEAN_REF = DATA_MEAN_DN / 10000.0
DATA_STD_REF = DATA_STD_DN / 10000.0


class PrithviEncoder:
    """
    Service wrapper around IBM-NASA Prithvi-EO-2.0-300M foundation model.
    Encodes multi-spectral patches into 1024-dimensional normalized vector embeddings.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        if device is None:
            env_dev = os.getenv("DEVICE")
            if env_dev:
                self.device = torch.device(env_dev)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing Prithvi-EO-2.0-300M on device '{self.device}'...")

        self.checkpoint_path = checkpoint_path or os.getenv("PRITHVI_CHECKPOINT_PATH")
        self._load_model()
        logger.info("Prithvi-EO-2.0-300M encoder successfully initialized!")

    def _load_model(self):
        """Loads PrithviMAE architecture and weights, downloading from HuggingFace if needed."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        local_cfg = os.path.join(base_dir, "models", "retrieval", "prithvi", CONFIG_FILENAME)
        local_mae = os.path.join(base_dir, "models", "retrieval", "prithvi", MAE_FILENAME)

        if os.path.exists(local_cfg) and os.path.exists(local_mae):
            logger.info("Loading Prithvi configuration and architecture from local disk...")
            cfg_file = local_cfg
            mae_file = local_mae
        else:
            from huggingface_hub import hf_hub_download
            logger.info(f"Loading Prithvi configuration from '{MODEL_REPO}'...")
            cfg_file = hf_hub_download(repo_id=MODEL_REPO, filename=CONFIG_FILENAME)
            mae_file = hf_hub_download(repo_id=MODEL_REPO, filename=MAE_FILENAME)

        # Dynamically import prithvi_mae
        spec = importlib.util.spec_from_file_location("prithvi_mae", mae_file)
        mae_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mae_module)

        # Patch device mismatch in HuggingFace prithvi_mae pos_embed interpolation
        orig_interpolate = mae_module._interpolate_pos_encoding
        def patched_interpolate(*args, **kwargs):
            pos_embed = args[0] if len(args) > 0 else kwargs.get("pos_embed")
            res = orig_interpolate(*args, **kwargs)
            if hasattr(pos_embed, "device") and res.device != pos_embed.device:
                return res.to(pos_embed.device)
            return res
        mae_module._interpolate_pos_encoding = patched_interpolate

        with open(cfg_file, "r") as f:
            full_cfg = json.load(f)
            pretrained_cfg = full_cfg.get("pretrained_cfg", full_cfg)

        # Configure model initialization (num_frames defaults to 4 in checkpoint)
        pretrained_cfg.update(
            in_chans=6,
            coords_encoding=[]
        )

        self.model = mae_module.PrithviMAE(**pretrained_cfg)

        # 2. Load weights
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            local_cand = os.path.join(base_dir, "models", "retrieval", "prithvi", CHECKPOINT_FILENAME)
            if os.path.exists(local_cand):
                self.checkpoint_path = local_cand
            else:
                logger.info(f"Downloading checkpoint '{CHECKPOINT_FILENAME}' from HuggingFace '{MODEL_REPO}'...")
                self.checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILENAME)

        logger.info(f"Loading weights from {self.checkpoint_path}...")
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()

    def _prepare_patch_tensor(self, crop: np.ndarray, target_size: int = 224) -> torch.Tensor:
        """
        Converts a multi-band patch crop (C, H, W) to standardized Prithvi input tensor (6, 1, 224, 224).
        """
        # Ensure 3D (C, H, W)
        if crop.ndim == 2:
            crop = crop[np.newaxis, :, :]

        c, h, w = crop.shape

        # 1. Band Mapping to 6 Prithvi Bands (B02, B03, B04, B05/B08, B06/B11, B07/B12)
        if c == 6:
            bands_data = crop[:6].astype(np.float32)
        elif c >= 6:
            bands_data = crop[:6].astype(np.float32)
        elif c == 4:
            # 4 bands: [Blue, Green, Red, NIR]. Synthesize SWIR1 and SWIR2
            b_b = crop[0]
            b_g = crop[1]
            b_r = crop[2]
            b_nir = crop[3]
            # Approximate SWIR1/SWIR2 from NIR and Red ratios
            b_swir1 = (b_nir * 0.8 + b_r * 0.2).astype(np.float32)
            b_swir2 = (b_nir * 0.5 + b_r * 0.5).astype(np.float32)
            bands_data = np.stack([b_b, b_g, b_r, b_nir, b_swir1, b_swir2], axis=0).astype(np.float32)
        elif c == 3:
            # 3 bands: RGB -> replicate into 6 channels
            b_b = crop[0]
            b_g = crop[1]
            b_r = crop[2]
            bands_data = np.stack([b_b, b_g, b_r, b_r, b_g, b_b], axis=0).astype(np.float32)
        else:
            # Replicate first band to reach 6
            replicated = [crop[i % c] for i in range(6)]
            bands_data = np.stack(replicated, axis=0).astype(np.float32)

        # 2. Reflectance vs DN Normalization
        # If values are in reflectance range [0.0, 1.5]
        max_val = np.nanmax(bands_data)
        if max_val <= 10.0:
            mean = DATA_MEAN_REF[:, np.newaxis, np.newaxis]
            std = DATA_STD_REF[:, np.newaxis, np.newaxis]
        else:
            mean = DATA_MEAN_DN[:, np.newaxis, np.newaxis]
            std = DATA_STD_DN[:, np.newaxis, np.newaxis]

        # Handle nan / inf
        bands_data = np.nan_to_num(bands_data, nan=0.0, posinf=1.5, neginf=0.0)
        norm_data = (bands_data - mean) / std

        # 3. Spatial Resampling to 224x224 (Bilinear)
        tensor = torch.from_numpy(norm_data).unsqueeze(0)  # (1, 6, H, W)
        if h != target_size or w != target_size:
            tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)

        # Add time dimension: (6, 1, 224, 224)
        tensor = tensor.squeeze(0).unsqueeze(1)
        return tensor

    def encode_multiband_patches(
        self,
        crops: List[np.ndarray],
        batch_size: int = 16,
        target_size: int = 224
    ) -> np.ndarray:
        """
        Encode a list of multi-spectral patch crops (C, H, W) into 1024-dimensional
        L2-normalized latent feature vectors.

        Args:
            crops: List of numpy arrays, each of shape (C, H, W).
            batch_size: Batch size for forward pass.
            target_size: Input spatial size (default 224).

        Returns:
            np.ndarray of shape (N, 1024), float32.
        """
        if not crops:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            batch_tensors = [self._prepare_patch_tensor(c, target_size) for c in batch_crops]
            # Shape: (B, 6, 1, 224, 224)
            batch_input = torch.stack(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                # Forward pass through Prithvi encoder
                features = self.model.encoder.forward_features(batch_input)
                # Last layer feature tokens: (B, 1 + num_patches, 1024)
                last_layer = features[-1]

                # Mean pool over spatial patch tokens (excluding CLS token at index 0)
                patch_tokens = last_layer[:, 1:, :]
                emb = patch_tokens.mean(dim=1)

                # L2 normalize
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embeddings.append(emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


_global_prithvi_encoder: Optional[PrithviEncoder] = None


def get_prithvi_encoder() -> PrithviEncoder:
    """Singleton getter for Prithvi-EO-2.0-300M encoder."""
    global _global_prithvi_encoder
    if _global_prithvi_encoder is None:
        _global_prithvi_encoder = PrithviEncoder()
    return _global_prithvi_encoder
