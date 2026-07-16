"""
data.py
=======
SuStaIn Dataset — MRI-only classification and staging.
Matches subject T1-MRIs to SuStaIn subtype probabilities and stage targets.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

TARGET_SHAPE = (128, 128, 128)


def resize_volume(volume: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    """
    Resize a 3D volume to target_shape using PyTorch trilinear interpolation.
    """
    if volume.shape == target_shape:
        return volume.astype(np.float32)
    tensor  = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=True)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


class SuStaInDataset(Dataset):
    """
    Loads MRI volumes from .npz files and pairs them with SuStaIn subtype probabilities
    and assigned stages from the sustain_subject_staging_results.csv file.
    """

    def __init__(self, npz_root: str, csv_path: str, transform=None):
        self.npz_root  = npz_root
        self.transform = transform

        # ── 1. Load the CSV and build label maps ─────────────────────────────
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        required_cols = ['PTID', 'Assigned_Subtype', 'Assigned_Stage', 'Prob_Subtype_1', 'Prob_Subtype_2', 'Prob_Subtype_3']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV file must contain column: {col}")

        # Drop any rows with missing essential staging info
        df = df.dropna(subset=['Assigned_Subtype', 'Assigned_Stage'])

        self._labels_map = {}
        for _, row in df.iterrows():
            ptid = str(row['PTID']).strip()
            self._labels_map[ptid] = {
                'assigned_subtype': int(row['Assigned_Subtype']) - 1,  # convert 1,2,3 -> 0,1,2
                'assigned_stage': float(row['Assigned_Stage']),
                'subtype_probs': np.array([
                    row['Prob_Subtype_1'],
                    row['Prob_Subtype_2'],
                    row['Prob_Subtype_3']
                ], dtype=np.float32)
            }

        # ── 2. Scan the npz directory and keep only matched subjects ─────────
        all_files = sorted([f for f in os.listdir(npz_root) if f.endswith('.npz')])

        self._subjects = []
        skipped = 0
        for fname in all_files:
            ptid = fname.replace('.npz', '')
            if ptid in self._labels_map:
                self._subjects.append((fname, ptid, self._labels_map[ptid]))
            else:
                skipped += 1

        print(f"[SuStaInDataset] Found {len(all_files)} .npz files in root.")
        print(f"[SuStaInDataset] Matched {len(self._subjects)} subjects to staging labels.")
        print(f"[SuStaInDataset] Skipped {skipped} subjects (not in staging results CSV).")

        # Cache targets as numpy array for stratified KFold splits (based on hard subtype label)
        self._cached_subtypes = np.array([info['assigned_subtype'] for _, _, info in self._subjects], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._subjects)

    def get_labels(self) -> np.ndarray:
        """Return hard subtype labels (used for stratified splits in train)."""
        return self._cached_subtypes

    def __getitem__(self, idx: int) -> dict:
        fname, ptid, label_info = self._subjects[idx]

        # ── Load .npz and extract MRI volume ─────────────────────────────────
        sample = np.load(os.path.join(self.npz_root, fname))
        mri    = sample['mwp1']

        # ── Safety: replace NaN/Inf with 0 ───────────────────────────────────
        mri = np.nan_to_num(mri, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Resize to 128³ ────────────────────────────────────────────────────
        mri = resize_volume(mri, TARGET_SHAPE)

        # ── Add channel dim: (1, 128, 128, 128) ──────────────────────────────
        mri = np.expand_dims(mri, axis=0)

        if self.transform:
            mri = self.transform(mri)

        return {
            'mri':              torch.from_numpy(mri),
            'subtype_probs':    torch.tensor(label_info['subtype_probs'], dtype=torch.float32),
            'assigned_subtype': torch.tensor(label_info['assigned_subtype'], dtype=torch.long),
            'assigned_stage':   torch.tensor(label_info['assigned_stage'], dtype=torch.float32),
            'subject_id':       ptid
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mock Dataset (for local pipeline verification)
# ─────────────────────────────────────────────────────────────────────────────

class MockDataset(Dataset):
    """
    Generates random 3D MRI volumes and mock SuStaIn labels for quick pipeline tests.
    """

    def __init__(self, size: int = 40):
        self.size = size
        
        rng = np.random.default_rng(42)
        # Random assigned subtypes 0, 1, 2
        self._subtypes = rng.choice([0, 1, 2], size=size)
        # Random assigned stages between 0 and 48
        self._stages = rng.uniform(0.0, 48.0, size=size).astype(np.float32)
        
        # Construct soft subtype probabilities corresponding roughly to the assigned subtype
        self._probs = []
        for s in self._subtypes:
            p = rng.dirichlet([1.0, 1.0, 1.0])
            p[s] += 2.0  # boost probability of the selected subtype
            p /= p.sum()
            self._probs.append(p)
        self._probs = np.array(self._probs, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def get_labels(self) -> np.ndarray:
        return self._subtypes

    def __getitem__(self, idx: int) -> dict:
        return {
            'mri':              torch.randn(1, *TARGET_SHAPE),
            'subtype_probs':    torch.tensor(self._probs[idx], dtype=torch.float32),
            'assigned_subtype': torch.tensor(self._subtypes[idx], dtype=torch.long),
            'assigned_stage':   torch.tensor(self._stages[idx], dtype=torch.float32),
            'subject_id':       f'mock_{idx:03d}'
        }
