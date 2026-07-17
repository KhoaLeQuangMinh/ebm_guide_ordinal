"""
model.py
========
SuStaIn multi-head classifier model.
Uses 3D ResNet-18 backbone (prototype-free) as encoder, followed by:
  - 1 subtype prediction head (size 3 output)
  - 3 subtype-specific stage prediction heads (size 48 output for ordinal classification)
"""

import torch
import torch.nn as nn
from src.models.resnet3d import resnet18_3d


class SuStaInCNN(nn.Module):
    """
    Multi-head 3D CNN model for SuStaIn progression modeling.
    Extracts 128-dimensional latent vector from MRI, then outputs:
      1. Subtype Logits: prediction over 3 subtype classes
      2. Stage Logits: 3 heads of shape (B, 48) containing event occurrence predictions
    """

    def __init__(self, feature_dim: int = 128, num_subtypes: int = 3, num_events: int = 48, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()

        # ── 1. Encoder Backbone ──────────────────────────────────────────────
        self.backbone = resnet18_3d()

        # ── 2. Subtype Predictor Head ────────────────────────────────────────
        # Two-layer MLP: 128 -> 64 -> 3
        self.subtype_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_subtypes)
        )

        # ── 3. Stage Predictor Heads ─────────────────────────────────────────
        # 3 independent MLPs (one for each subtype) mapping 128 -> 64 -> 48
        # Each outputs cumulative logits for 48 sequential events.
        self.stage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_events)
            ) for _ in range(num_subtypes)
        ])

        # ── 4. Clinical Diagnosis Predictor Head ─────────────────────────────
        # Two-layer MLP: 128 -> 64 -> 4 (CN, sMCI, pMCI, AD)
        self.diag_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, mri: torch.Tensor):
        """
        Parameters
        ----------
        mri : (B, 1, 128, 128, 128)

        Returns
        -------
        subtype_logits : (B, 3) - raw logits for subtype classification
        stage_outputs  : list of 3 tensors, each of shape (B, 48) - event progression logits per head
        diag_logits    : (B, 4) - raw logits for 4-class clinical diagnosis
        """
        features = self.backbone(mri)            # (B, 128)
        subtype_logits = self.subtype_head(features)  # (B, 3)
        stage_outputs = [head(features) for head in self.stage_heads]  # 3 x (B, 48)
        diag_logits = self.diag_head(features)    # (B, 4)
        return subtype_logits, stage_outputs, diag_logits
