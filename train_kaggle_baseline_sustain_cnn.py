"""
train_kaggle_baseline_sustain_cnn.py
====================================
SuStaIn 3D ResNet-18 Multi-Task CNN with FIX 1 (Masked Global Pooling).

Fix 1 Upgrades:
  1. Layer 4 Stride = 1: Preserves 8x8x8 spatial resolution (512 spatial feature cells).
  2. Masked Global Pooling: Dynamically downsamples 3D brain mask to 8x8x8 and zeroes out
     100% of background BatchNorm noise before feature projection to 128D latent vector.

All other components (5-fold CV, loss functions, CLI arguments, OOF CSV export) remain identical.
"""

import argparse
import copy
import csv
import datetime
import json
import math
import os
import random
import sys
import warnings
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', message='.*deterministic.*')


# ══════════════════════════════════════════════════════════════════════════════
# 1. UTILS (src/utils.py)
# ══════════════════════════════════════════════════════════════════════════════

class Tee:
    def __init__(self, file_handle):
        self._file = file_handle
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def restore(self):
        return self._stdout


def save_run_config(args, log_path: str) -> "Tee":
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a")

    header = [
        "=" * 62,
        f"  RUN  —  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        "=" * 62,
        "  ARGUMENTS",
        "-" * 62,
    ]
    for k, v in vars(args).items():
        header.append(f"  {k:<30} {v}")
    header += ["-" * 62, "  TRAINING LOG", "-" * 62, ""]

    block = "\n".join(header) + "\n"
    log_file.write(block)
    log_file.flush()

    tee = Tee(log_file)
    sys.stdout = tee
    return tee


def print_experiment_config(args):
    print("\n" + "=" * 60)
    print(f"  Experiment : {args.experiment_name}")
    print("-" * 60)
    print(f"  Task       : SuStaIn Subtype & Stage Multi-Task CNN")
    print(f"  Encoder    : 3D ResNet-18 + Masked Global Pooling (Fix 1)")
    print(f"  Resolution : Layer 4 (8x8x8 = 512 spatial cells)")
    print(f"  Loss       : Soft Cross-Entropy (Subtype) + Ordinal BCE (Stage)")
    print("-" * 60)
    print(f"  Device     : {args.device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch Size : {args.batch_size}")
    print("-" * 60)
    print(f"  LR         : {args.lr}")
    print(f"  Weight dec.: {args.weight_decay}")
    print(f"  Gamma (Dec): {args.gamma}")
    print("=" * 60 + "\n")


def set_global_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 3D RESNET-18 BACKBONE (Layer 4 Stride=1 for 8x8x8 Resolution)
# ══════════════════════════════════════════════════════════════════════════════

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4),
        device=out.device, dtype=out.dtype,
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        shortcut_type='B',
    ):
        self.inplanes = 64
        super().__init__()

        self.conv1 = nn.Conv3d(
            1, 64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        # FIX 1: Set Layer 4 stride=1 to maintain 8x8x8 spatial resolution (512 spatial cells)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_maps = self.layer4(x)  # Shape: (B, 512, 8, 8, 8)
        return feature_maps


def resnet18_3d(**kwargs) -> ResNet3D:
    return ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MASKED GLOBAL POOLING & SUSTAIN CNN MODEL (FIX 1)
# ══════════════════════════════════════════════════════════════════════════════

def generate_downsampled_mask(x: torch.Tensor, target_shape, threshold: float = 1e-4) -> torch.Tensor:
    """
    Dynamically generates downsampled 3D binary brain mask M (B, 1, D, H, W) matching target_shape.
    1. Threshold raw MRI volume (> 1e-4) -> binary mask.
    2. Apply 3D Max-Pooling (kernel=3, padding=1) to dilate and smoothly pad outer brain shell.
    3. Adaptively downsample to target_shape (e.g. 8x8x8) using F.adaptive_max_pool3d.
    """
    raw_mask = (x > threshold).float()
    dilated_mask = F.max_pool3d(raw_mask, kernel_size=3, stride=1, padding=1)
    mask_down = F.adaptive_max_pool3d(dilated_mask, output_size=target_shape)
    return mask_down


class SuStaInCNN(nn.Module):
    def __init__(self, feature_dim: int = 128, num_subtypes: int = 3, num_events: int = 48, dropout: float = 0.1):
        super().__init__()

        self.backbone = resnet18_3d()

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, feature_dim),
        )

        self.subtype_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_subtypes)
        )

        self.stage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_events)
            ) for _ in range(num_subtypes)
        ])

    def forward(self, mri: torch.Tensor):
        # 1. Extract 3D feature map from backbone (B, 512, 8, 8, 8)
        feature_map = self.backbone(mri)

        # 2. FIX 1: Generate downsampled 8x8x8 3D binary brain mask
        mask = generate_downsampled_mask(mri, target_shape=feature_map.shape[2:]).to(mri.device)

        # 3. FIX 1: Zero out background feature noise
        masked_map = feature_map * mask

        # 4. FIX 1: Masked Global Average Pooling (average ONLY valid brain feature cells)
        pooled = masked_map.sum(dim=(2, 3, 4)) / (mask.sum(dim=(2, 3, 4)) + 1e-8)  # (B, 512)

        # 5. Project to 128D latent vector
        features = self.fc1(pooled)  # (B, 128)

        subtype_logits = self.subtype_head(features)
        stage_outputs = [head(features) for head in self.stage_heads]
        return subtype_logits, stage_outputs


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET (src/data.py)
# ══════════════════════════════════════════════════════════════════════════════

TARGET_SHAPE = (128, 128, 128)

def resize_volume(volume: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    if volume.shape == target_shape:
        return volume.astype(np.float32)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=True)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


class SuStaInDataset(Dataset):
    def __init__(self, npz_root: str, csv_path: str, transform=None):
        self.npz_root = npz_root
        self.transform = transform

        df = pd.read_csv(csv_path)
        required_cols = ['PTID', 'Assigned_Subtype', 'Assigned_Stage', 'Prob_Subtype_1', 'Prob_Subtype_2', 'Prob_Subtype_3']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV file must contain column: {col}")

        df = df.dropna(subset=['Assigned_Subtype', 'Assigned_Stage'])

        self._labels_map = {}
        for _, row in df.iterrows():
            ptid = str(row['PTID']).strip()
            self._labels_map[ptid] = {
                'assigned_subtype': int(row['Assigned_Subtype']) - 1,
                'assigned_stage': float(row['Assigned_Stage']),
                'subtype_probs': np.array([
                    row['Prob_Subtype_1'],
                    row['Prob_Subtype_2'],
                    row['Prob_Subtype_3']
                ], dtype=np.float32)
            }

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

        self._cached_subtypes = np.array([info['assigned_subtype'] for _, _, info in self._subjects], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._subjects)

    def get_labels(self) -> np.ndarray:
        return self._cached_subtypes

    def __getitem__(self, idx: int) -> dict:
        fname, ptid, label_info = self._subjects[idx]

        sample = np.load(os.path.join(self.npz_root, fname))
        mri = sample['mwp1'] if 'mwp1' in sample else (sample['mri'] if 'mri' in sample else sample['image'])
        mri = np.nan_to_num(mri, nan=0.0, posinf=0.0, neginf=0.0)
        mri = resize_volume(mri, TARGET_SHAPE)
        mri = np.expand_dims(mri, axis=0)

        if self.transform:
            mri = self.transform(mri)

        return {
            'mri': torch.from_numpy(mri),
            'subtype_probs': torch.tensor(label_info['subtype_probs'], dtype=torch.float32),
            'assigned_subtype': torch.tensor(label_info['assigned_subtype'], dtype=torch.long),
            'assigned_stage': torch.tensor(label_info['assigned_stage'], dtype=torch.float32),
            'subject_id': ptid
        }


class MockDataset(Dataset):
    def __init__(self, size: int = 40):
        self.size = size
        rng = np.random.default_rng(42)
        self._subtypes = rng.choice([0, 1, 2], size=size)
        self._stages = rng.uniform(0.0, 48.0, size=size).astype(np.float32)

        self._probs = []
        for s in self._subtypes:
            p = rng.dirichlet([1.0, 1.0, 1.0])
            p[s] += 2.0
            p /= p.sum()
            self._probs.append(p)
        self._probs = np.array(self._probs, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def get_labels(self) -> np.ndarray:
        return self._subtypes

    def __getitem__(self, idx: int) -> dict:
        return {
            'mri': torch.randn(1, *TARGET_SHAPE),
            'subtype_probs': torch.tensor(self._probs[idx], dtype=torch.float32),
            'assigned_subtype': torch.tensor(self._subtypes[idx], dtype=torch.long),
            'assigned_stage': torch.tensor(self._stages[idx], dtype=torch.float32),
            'subject_id': f'mock_{idx:03d}'
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING & EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='SuStaIn CNN with FIX 1 (Masked Global Pooling): Train 3D ResNet-18 for Subtype and Stage.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--experiment_name', type=str, required=True,
                   help='Unique name for the run.')
    p.add_argument('--mock_data', action='store_true',
                   help='Use randomly generated mock tensors for local pipeline testing.')
    p.add_argument('--data_root', type=str, default='/kaggle/input/kisokoghan-paired-npz/paired_npz',
                   help='Directory containing subject .npz files.')
    p.add_argument('--csv_path', type=str, default='/kaggle/input/sustain-data/sustain_subject_staging_results_filter_cn.csv',
                   help='Path to the sustain staging results CSV file.')
    p.add_argument('--train_ratio', type=float, default=0.7,
                   help='Train fraction for single-split mode.')
    p.add_argument('--val_ratio', type=float, default=0.1,
                   help='Validation fraction for single-split mode.')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--kfold', type=int, default=5,
                   help='Number of cross-validation folds. 0 = disabled.')
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.95,
                   help='LR decay rate for ExponentialLR.')
    p.add_argument('--lambda_subtype', type=float, default=1.0)
    p.add_argument('--lambda_stage', type=float, default=1.0)

    args = p.parse_args()
    return args


def train_epoch(model, loader, optimizer, args):
    model.train()
    total_loss = 0.0
    total_loss_sub = 0.0
    total_loss_stage = 0.0

    for batch in loader:
        mri = batch['mri'].to(args.device)
        subtype_probs = batch['subtype_probs'].to(args.device)
        assigned_subtype = batch['assigned_subtype'].to(args.device)
        assigned_stage = batch['assigned_stage'].to(args.device)

        B = mri.size(0)
        subtype_logits, stage_logits_list = model(mri)

        log_probs = F.log_softmax(subtype_logits, dim=-1)
        loss_sub = -(subtype_probs * log_probs).sum(dim=-1).mean()

        ordinal_targets = torch.zeros(B, 48, device=args.device)
        for i in range(B):
            S = int(torch.clamp(assigned_stage[i], min=0, max=48).item())
            ordinal_targets[i, :S] = 1.0

        stacked_stage_logits = torch.stack(stage_logits_list, dim=1)
        batch_indices = torch.arange(B, device=args.device)
        assigned_logits = stacked_stage_logits[batch_indices, assigned_subtype, :]

        loss_stage = F.binary_cross_entropy_with_logits(assigned_logits, ordinal_targets)

        loss = args.lambda_subtype * loss_sub + args.lambda_stage * loss_stage

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_loss_sub += loss_sub.item() * B
        total_loss_stage += loss_stage.item() * B

    num_samples = len(loader.dataset)
    return total_loss / num_samples, total_loss_sub / num_samples, total_loss_stage / num_samples


def evaluate(model, loader, args):
    model.eval()
    total_loss = 0.0
    total_loss_sub = 0.0
    total_loss_stage = 0.0

    all_sub_preds = []
    all_sub_targets = []
    all_sub_probs_pred = []
    all_sub_probs_target = []
    all_stage_preds = []
    all_stage_targets = []

    with torch.no_grad():
        for batch in loader:
            mri = batch['mri'].to(args.device)
            subtype_probs = batch['subtype_probs'].to(args.device)
            assigned_subtype = batch['assigned_subtype'].to(args.device)
            assigned_stage = batch['assigned_stage'].to(args.device)

            B = mri.size(0)
            subtype_logits, stage_logits_list = model(mri)

            log_probs = F.log_softmax(subtype_logits, dim=-1)
            loss_sub = -(subtype_probs * log_probs).sum(dim=-1).mean()

            ordinal_targets = torch.zeros(B, 48, device=args.device)
            for i in range(B):
                S = int(torch.clamp(assigned_stage[i], min=0, max=48).item())
                ordinal_targets[i, :S] = 1.0

            stacked_stage_logits = torch.stack(stage_logits_list, dim=1)
            batch_indices = torch.arange(B, device=args.device)
            assigned_logits = stacked_stage_logits[batch_indices, assigned_subtype, :]

            loss_stage = F.binary_cross_entropy_with_logits(assigned_logits, ordinal_targets)
            loss = args.lambda_subtype * loss_sub + args.lambda_stage * loss_stage

            total_loss += loss.item() * B
            total_loss_sub += loss_sub.item() * B
            total_loss_stage += loss_stage.item() * B

            pred_sub = torch.argmax(subtype_logits, dim=-1)
            all_sub_preds.extend(pred_sub.cpu().numpy())
            all_sub_targets.extend(assigned_subtype.cpu().numpy())

            probs_sub_pred = torch.softmax(subtype_logits, dim=-1)
            all_sub_probs_pred.extend(probs_sub_pred.cpu().numpy())
            all_sub_probs_target.extend(subtype_probs.cpu().numpy())

            reconstructed_stages = torch.zeros(B, device=args.device)
            for c in range(3):
                probs_event_c = torch.sigmoid(stage_logits_list[c])
                stage_c = probs_event_c.sum(dim=-1)
                reconstructed_stages += probs_sub_pred[:, c] * stage_c

            all_stage_preds.extend(reconstructed_stages.cpu().numpy())
            all_stage_targets.extend(assigned_stage.cpu().numpy())

    num_samples = len(loader.dataset)
    avg_loss = total_loss / num_samples
    avg_loss_sub = total_loss_sub / num_samples
    avg_loss_stage = total_loss_stage / num_samples

    acc = accuracy_score(all_sub_targets, all_sub_preds)
    f1 = f1_score(all_sub_targets, all_sub_preds, average='macro')
    sub_prob_mse = np.mean((np.array(all_sub_probs_pred) - np.array(all_sub_probs_target))**2)
    mae = np.mean(np.abs(np.array(all_stage_preds) - np.array(all_stage_targets)))

    pred_stages_rounded = np.clip(np.round(all_stage_preds), 0, 48).astype(int)
    true_stages_rounded = np.clip(np.round(all_stage_targets), 0, 48).astype(int)
    qwk = cohen_kappa_score(true_stages_rounded, pred_stages_rounded, weights='quadratic', labels=list(range(49)))

    rho, _ = spearmanr(all_stage_preds, all_stage_targets)
    if np.isnan(rho):
        rho = 0.0

    return avg_loss, avg_loss_sub, avg_loss_stage, acc, f1, sub_prob_mse, mae, qwk, rho


def predict_test(model, loader, args):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            mri = batch['mri'].to(args.device)
            subtype_probs = batch['subtype_probs'].to(args.device)
            assigned_subtype = batch['assigned_subtype'].to(args.device)
            assigned_stage = batch['assigned_stage'].to(args.device)
            subject_ids = batch['subject_id']

            subtype_logits, stage_logits_list = model(mri)

            pred_sub = torch.argmax(subtype_logits, dim=-1).cpu().numpy() + 1
            probs_sub_pred = torch.softmax(subtype_logits, dim=-1).cpu().numpy()

            reconstructed_stages = torch.zeros(mri.size(0), device=args.device)
            probs_sub_pred_torch = torch.softmax(subtype_logits, dim=-1)
            for c in range(3):
                probs_event_c = torch.sigmoid(stage_logits_list[c])
                stage_c = probs_event_c.sum(dim=-1)
                reconstructed_stages += probs_sub_pred_torch[:, c] * stage_c
            pred_stages = reconstructed_stages.cpu().numpy()

            for i in range(mri.size(0)):
                predictions.append({
                    'PTID': subject_ids[i],
                    'Assigned_Subtype_True': int(assigned_subtype[i].item()) + 1,
                    'Assigned_Subtype_Pred': int(pred_sub[i]),
                    'Assigned_Stage_True': float(assigned_stage[i].item()),
                    'Assigned_Stage_Pred': float(pred_stages[i]),
                    'Prob_Subtype_1_True': float(subtype_probs[i, 0].item()),
                    'Prob_Subtype_2_True': float(subtype_probs[i, 1].item()),
                    'Prob_Subtype_3_True': float(subtype_probs[i, 2].item()),
                    'Prob_Subtype_1_Pred': float(probs_sub_pred[i, 0]),
                    'Prob_Subtype_2_Pred': float(probs_sub_pred[i, 1]),
                    'Prob_Subtype_3_Pred': float(probs_sub_pred[i, 2]),
                })
    return predictions


def train_fold_pipeline(train_idx, val_idx, test_idx, dataset, fold_name, args):
    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)
    test_sub = Subset(dataset, test_idx)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SuStaInCNN().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_qwk = -1.0
    best_val_rho = 0.0

    output_dir = os.path.join("outputs", "runs", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_sub, tr_stg = train_epoch(model, train_loader, optimizer, args)
        val_loss, val_sub, val_stg, val_acc, val_f1, val_mse, val_mae, val_qwk, val_rho = evaluate(model, val_loader, args)

        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"Tr Loss: {tr_loss:.4f} (Sub: {tr_sub:.3f}, Stg: {tr_stg:.3f}) | "
                  f"Val Loss: {val_loss:.4f} (Sub: {val_sub:.3f}, Stg: {val_stg:.3f}) | "
                  f"Sub Acc: {val_acc:.3f}, F1: {val_f1:.3f}, MSE: {val_mse:.4f} | "
                  f"Stg MAE: {val_mae:.2f}, QWK: {val_qwk:.3f}, Rho: {val_rho:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_val_qwk = val_qwk
            best_val_rho = val_rho
            best_path = os.path.join(output_dir, f"best_model_{fold_name}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_qwk': val_qwk,
                'val_rho': val_rho
            }, best_path)

    latest_path = os.path.join(output_dir, f"latest_model_{fold_name}.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, latest_path)

    print(f"Loading best model checkpoint for Fold {fold_name} and running inference on Test partition...")
    checkpoint = torch.load(os.path.join(output_dir, f"best_model_{fold_name}.pth"), map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_sub, test_stg, test_acc, test_f1, test_mse, test_mae, test_qwk, test_rho = evaluate(model, test_loader, args)
    print(f"---> Fold {fold_name} TEST Performance: Loss: {test_loss:.4f} | Sub Acc: {test_acc:.3f}, F1: {test_f1:.3f} | Stg MAE: {test_mae:.2f}, QWK: {test_qwk:.3f}")

    test_predictions = predict_test(model, test_loader, args)
    for pred in test_predictions:
        pred['Test_Fold'] = fold_name

    return test_loss, test_acc, test_f1, test_mae, test_qwk, test_rho, test_predictions


def main():
    args = parse_args()
    set_global_seed(args.seed)

    log_dir = os.path.join("outputs", "runs", args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    tee = save_run_config(args, os.path.join(log_dir, "train_log.txt"))

    print_experiment_config(args)

    if args.mock_data:
        print("Using MOCK data for local verification...")
        dataset = MockDataset(size=40)
    else:
        print(f"Loading real data from: {args.data_root}")
        dataset = SuStaInDataset(npz_root=args.data_root, csv_path=args.csv_path)

    all_oof_predictions = []

    if args.kfold > 0:
        print(f"Starting {args.kfold}-Fold Stratified Cross Validation...")
        subtypes = dataset.get_labels()
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

        test_losses, test_accs, test_f1s, test_maes, test_qwks, test_rhos = [], [], [], [], [], []

        for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), subtypes), start=1):
            rng = np.random.default_rng(args.seed + fold)
            shuffled_trainval = rng.permutation(trainval_idx)
            split_point = int(0.85 * len(shuffled_trainval))
            train_idx = shuffled_trainval[:split_point]
            val_idx = shuffled_trainval[split_point:]

            loss, acc, f1, mae, qwk, rho, preds = train_fold_pipeline(
                train_idx, val_idx, test_idx, dataset, f"fold{fold}", args
            )

            test_losses.append(loss)
            test_accs.append(acc)
            test_f1s.append(f1)
            test_maes.append(mae)
            test_qwks.append(qwk)
            test_rhos.append(rho)
            all_oof_predictions.extend(preds)

        print(f"\n==================================================")
        print(f"OOF (Out-Of-Fold) Test Summary ({args.kfold} Folds):")
        print(f"  Mean Test Loss: {np.mean(test_losses):.4f} +/- {np.std(test_losses):.4f}")
        print(f"  Mean Test Acc:  {np.mean(test_accs):.3f} +/- {np.std(test_accs):.3f}")
        print(f"  Mean Test F1:   {np.mean(test_f1s):.3f} +/- {np.std(test_f1s):.3f}")
        print(f"  Mean Test MAE:  {np.mean(test_maes):.2f} +/- {np.std(test_maes):.2f}")
        print(f"  Mean Test QWK:  {np.mean(test_qwks):.3f} +/- {np.std(test_qwks):.3f}")
        print(f"  Mean Test Rho:  {np.mean(test_rhos):.3f} +/- {np.std(test_rhos):.3f}")
        print(f"==================================================")
    else:
        print("Starting Single Split Train/Val/Test...")
        generator = torch.Generator().manual_seed(args.seed)
        train_size = int(args.train_ratio * len(dataset))
        val_size = int(args.val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        trainval_ds, test_ds = random_split(
            dataset, [train_size + val_size, test_size], generator=generator
        )

        train_size_split = int((args.train_ratio / (args.train_ratio + args.val_ratio)) * len(trainval_ds))
        val_size_split = len(trainval_ds) - train_size_split
        train_ds, val_ds = random_split(
            trainval_ds, [train_size_split, val_size_split], generator=generator
        )

        _, _, _, _, _, _, preds = train_fold_pipeline(
            train_ds.indices, val_ds.indices, test_ds.indices, dataset, "single_split", args
        )
        all_oof_predictions.extend(preds)

    pred_df = pd.DataFrame(all_oof_predictions)
    pred_csv_path = os.path.join(log_dir, f"{args.experiment_name}_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\n✓ Saved test predictions CSV to: {pred_csv_path}")

    sys.stdout = tee.restore()
    print("Training process completed.")


if __name__ == '__main__':
    main()
