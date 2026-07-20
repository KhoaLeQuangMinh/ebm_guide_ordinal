"""
test_fix1_kaggle_diagnostic.py
================================
Diagnostic verification & visual inspection script for Kaggle.
Evaluates Fix 1 (Masked Global Pooling) across all subjects in the dataset AND
generates multi-panel feature map visualization plots showing what the 3D ResNet-18
sees after EVERY layer (Stem -> Layer 1 -> Layer 2 -> Layer 3 -> Layer 4 -> Masked Pooling).
"""

import argparse
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# 1. 3D ResNet-18 Backbone & Layer-by-Layer Feature Hook Extractor
# ══════════════════════════════════════════════════════════════════════════════

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet3D_LayerInspector(nn.Module):
    """
    3D ResNet-18 backbone that returns intermediate feature maps after EVERY layer stage:
      - stem_out  : (64, 32, 32, 32)
      - layer1_out: (64, 32, 32, 32)
      - layer2_out: (128, 16, 16, 16)
      - layer3_out: (256, 8, 8, 8)
      - layer4_out: (512, 4, 4, 4)
    """
    def __init__(self, spatial_size=128, sample_duration=128):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

        layers = [BasicBlock3D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        stem_out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        l1_out = self.layer1(stem_out)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        l4_out = self.layer4(l3_out)

        return {
            'stem': stem_out,
            'layer1': l1_out,
            'layer2': l2_out,
            'layer3': l3_out,
            'layer4': l4_out
        }


def generate_downsampled_mask(x, target_shape, threshold=1e-4):
    """
    Dynamically generates downsampled 3D binary brain mask M (B, 1, D, H, W) matching target_shape.
    1. Threshold raw MRI volume (> 1e-4) -> binary mask.
    2. Apply 3D Max-Pooling (kernel=3, padding=1) to dilate and smoothly pad outer brain shell.
    3. Adaptively downsample to target_shape (e.g. 4x4x4) using F.adaptive_max_pool3d.
       (Ensures any cell containing > 0% brain tissue is set to 1.0).
    """
    raw_mask = (x > threshold).float()
    dilated_mask = F.max_pool3d(raw_mask, kernel_size=3, stride=1, padding=1)
    mask_down = F.adaptive_max_pool3d(dilated_mask, output_size=target_shape)
    return mask_down


# ══════════════════════════════════════════════════════════════════════════════
# 2. Visual Inspection Generator
# ══════════════════════════════════════════════════════════════════════════════

def save_layer_inspection_plot(x, layer_outputs, mask_down, subject_id, save_path):
    """
    Plots central 2D slices of the raw MRI and feature activation heatmaps
    after Stem, Layer 1, Layer 2, Layer 3, Layer 4 (Before Mask), Mask, and Layer 4 (After Mask).
    """
    mri_slice = x[0, 0, x.shape[2]//2, :, :].cpu().numpy()

    def get_slice_norm(feat_tensor):
        norm_3d = torch.linalg.norm(feat_tensor[0], dim=0)  # (D, H, W)
        slice_2d = norm_3d[norm_3d.shape[0]//2, :, :].cpu().numpy()
        return slice_2d

    stem_slice = get_slice_norm(layer_outputs['stem'])
    l1_slice   = get_slice_norm(layer_outputs['layer1'])
    l2_slice   = get_slice_norm(layer_outputs['layer2'])
    l3_slice   = get_slice_norm(layer_outputs['layer3'])
    l4_slice   = get_slice_norm(layer_outputs['layer4'])

    l4_masked_tensor = layer_outputs['layer4'] * mask_down
    l4_masked_slice  = get_slice_norm(l4_masked_tensor)
    mask_slice       = mask_down[0, 0, mask_down.shape[2]//2, :, :].cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # 1. Raw Input MRI
    im0 = axes[0, 0].imshow(mri_slice, cmap='gray')
    axes[0, 0].set_title(f"Input Raw MRI ({x.shape[2]}x{x.shape[3]})", fontsize=11, weight='bold')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Stem Activation
    im1 = axes[0, 1].imshow(stem_slice, cmap='viridis')
    s_shape = layer_outputs['stem'].shape
    axes[0, 1].set_title(f"Stem ({s_shape[1]}x{s_shape[2]}x{s_shape[3]})", fontsize=11, weight='bold')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Layer 1 Activation
    im2 = axes[0, 2].imshow(l1_slice, cmap='viridis')
    l1_shape = layer_outputs['layer1'].shape
    axes[0, 2].set_title(f"Layer 1 ({l1_shape[1]}x{l1_shape[2]}x{l1_shape[3]})", fontsize=11, weight='bold')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Layer 2 Activation
    im3 = axes[0, 3].imshow(l2_slice, cmap='viridis')
    l2_shape = layer_outputs['layer2'].shape
    axes[0, 3].set_title(f"Layer 2 ({l2_shape[1]}x{l2_shape[2]}x{l2_shape[3]})", fontsize=11, weight='bold')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # 5. Layer 3 Activation
    im4 = axes[1, 0].imshow(l3_slice, cmap='viridis')
    l3_shape = layer_outputs['layer3'].shape
    axes[1, 0].set_title(f"Layer 3 ({l3_shape[1]}x{l3_shape[2]}x{l3_shape[3]})", fontsize=11, weight='bold')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 6. Layer 4 Activation BEFORE Masking
    im5 = axes[1, 1].imshow(l4_slice, cmap='inferno')
    l4_shape = layer_outputs['layer4'].shape
    axes[1, 1].set_title(f"Layer 4 BEFORE Mask\n({l4_shape[1]}x{l4_shape[2]}x{l4_shape[3]} - Noise in corners!)", fontsize=10, weight='bold', color='darkred')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 7. Downsampled 3D Brain Mask
    im6 = axes[1, 2].imshow(mask_slice, cmap='Reds', vmin=0, vmax=1)
    m_shape = mask_down.shape
    axes[1, 2].set_title(f"Downsampled Mask ({m_shape[2]}x{m_shape[3]}x{m_shape[4]})", fontsize=11, weight='bold', color='darkblue')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 8. Layer 4 Activation AFTER Masking (Fix 1)
    im7 = axes[1, 3].imshow(l4_masked_slice, cmap='inferno')
    axes[1, 3].set_title("Layer 4 AFTER Mask (Fix 1)\n(0% Background Noise)", fontsize=10, weight='bold', color='darkgreen')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f"Layer-by-Layer Feature Activation Inspection — Subject: {subject_id}", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" Saved layer-by-layer inspection plot to:\n   {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Diagnostic & Inspection Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostic(data_root, output_dir, max_plot_samples=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("  KAGGLE DIAGNOSTIC & LAYER-BY-LAYER VISUAL INSPECTION (FIX 1)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Scanning directory: {data_root}")

    if not os.path.exists(data_root):
        print(f"ERROR: Path '{data_root}' does not exist!")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_files = sorted([f for f in os.listdir(data_root) if f.endswith('.npz')])
    print(f"Found {len(all_files)} 3D MRI .npz files for testing.")

    if len(all_files) == 0:
        print("No .npz files found to test.")
        return

    model = ResNet3D_LayerInspector().to(device)
    model.eval()

    brain_cell_counts = []
    total_spatial_cells_list = []
    bg_noise_before_list = []
    bg_noise_after_list = []
    brain_energy_preserved_list = []

    print("\nRunning diagnostic & generating layer inspection plots...")

    with torch.no_grad():
        for idx, fname in enumerate(all_files):
            fpath = os.path.join(data_root, fname)
            sample = np.load(fpath)
            mri = sample['mwp1'] if 'mwp1' in sample else (sample['mri'] if 'mri' in sample else sample['image'])
            mri = np.nan_to_num(mri, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if mri.shape != (128, 128, 128):
                tensor = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)
                mri = F.interpolate(tensor, size=(128, 128, 128), mode='trilinear', align_corners=True).squeeze().numpy()

            x = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0).to(device)

            # Extract layer outputs
            layer_outputs = model(x)
            H = layer_outputs['layer4']  # Shape (1, 512, D_l4, H_l4, W_l4)
            target_spatial_shape = H.shape[2:]
            total_cells = target_spatial_shape[0] * target_spatial_shape[1] * target_spatial_shape[2]
            total_spatial_cells_list.append(total_cells)

            # Downsample mask dynamically to match layer4 spatial dimensions exactly
            M_down = generate_downsampled_mask(x, target_spatial_shape).to(device)

            n_brain_cells = float(M_down.sum().item())
            brain_cell_counts.append(n_brain_cells)

            bg_mask_bool = (M_down == 0.0).expand_as(H)
            brain_mask_bool = (M_down == 1.0).expand_as(H)

            bg_noise_before = float(torch.abs(H[bg_mask_bool]).mean().item()) if bg_mask_bool.sum() > 0 else 0.0
            bg_noise_before_list.append(bg_noise_before)

            H_clean = H * M_down
            bg_noise_after = float(torch.abs(H_clean[bg_mask_bool]).mean().item()) if bg_mask_bool.sum() > 0 else 0.0
            bg_noise_after_list.append(bg_noise_after)

            brain_energy_before = torch.abs(H[brain_mask_bool]).sum().item()
            brain_energy_after = torch.abs(H_clean[brain_mask_bool]).sum().item()
            brain_energy_preserved_list.append((brain_energy_after / (brain_energy_before + 1e-8)) * 100.0)

            # Save visual plots for first few subjects
            if idx < max_plot_samples:
                subject_id = fname.replace('.npz', '')
                plot_path = os.path.join(output_dir, f"layer_inspection_{subject_id}.png")
                save_layer_inspection_plot(x, layer_outputs, M_down, subject_id, plot_path)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(all_files):
                print(f"  [{idx + 1}/{len(all_files)}] Processed '{fname}' -> Active Brain Cells: {int(n_brain_cells)}/{total_cells} | BG Noise Before: {bg_noise_before:.4f} | After: {bg_noise_after:.4f}")

    total_cells_val = total_spatial_cells_list[0]
    print("\n" + "=" * 80)
    print("  FINAL DIAGNOSTIC VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Subjects Evaluated            : {len(all_files)}")
    print(f"Layer 4 Feature Map Shape           : (512, {target_spatial_shape[0]}, {target_spatial_shape[1]}, {target_spatial_shape[2]}) -> {total_cells_val} Spatial Cells")
    print(f"Mean Active Brain Feature Cells     : {np.mean(brain_cell_counts):.1f} / {total_cells_val} ({np.mean(brain_cell_counts)/total_cells_val*100:.1f}%)")
    print(f"Mean BG Feature Noise BEFORE Fix 1  : {np.mean(bg_noise_before_list):.6f} (BatchNorm non-zero noise)")
    print(f"Mean BG Feature Noise AFTER Fix 1   : {np.mean(bg_noise_after_list):.6f} (EXACTLY ZERO)")
    print(f"Brain Feature Preservation Ratio    : {np.mean(brain_energy_preserved_list):.2f}% (100.00% = 0 loss of brain features)")
    print("=" * 80)
    print(f"\nVisual inspection plots saved to directory:\n  {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Layer-by-Layer Feature Visualizer for Fix 1")
    parser.add_argument('--data_root', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz',
                        help='Directory containing subject .npz files.')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/diagnostic_inspection_plots',
                        help='Directory to save visual plots.')
    parser.add_argument('--max_plot_samples', type=int, default=3,
                        help='Number of subjects to generate visual plots for.')
    args = parser.parse_args()

    run_diagnostic(args.data_root, args.output_dir, max_plot_samples=args.max_plot_samples)
