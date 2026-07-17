"""
train_sustain_cnn.py
====================
Training entry point for the SuStaIn CNN multi-head progression predictor.
Supports soft cross-entropy for subtype classification, ordinal BCE for stage
progression, and pure hard masking to isolate subtype trajectories.
Includes Train/Val/Test partitioning per fold and exports Out-Of-Fold test predictions.
"""

import argparse
import copy
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', message='.*deterministic.*')

from src.data import SuStaInDataset, MockDataset
from src.model import SuStaInCNN
from src.utils import set_global_seed, seed_worker, save_run_config, print_experiment_config


# ══════════════════════════════════════════════════════════════════════════════
# Argument Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='SuStaIn CNN: Train 3D ResNet-18 for Subtype and Stage prediction.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--experiment_name', type=str, required=True,
                   help='Unique name for the run.')
    p.add_argument('--mock_data', action='store_true',
                   help='Use randomly generated mock tensors for local pipeline testing.')
    p.add_argument('--data_root', type=str, default='/kaggle/input/kisokoghan-paired-npz/paired_npz',
                   help='Directory containing subject .npz files.')
    p.add_argument('--csv_path', type=str, default='/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results.csv',
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

    # Optimizer & Scheduler (matching HOPE options)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.95,
                   help='LR decay rate for ExponentialLR.')
    
    # Loss scaling weights
    p.add_argument('--lambda_subtype', type=float, default=1.0)
    p.add_argument('--lambda_stage', type=float, default=1.0)
    p.add_argument('--lambda_diag', type=float, default=1.0)

    args = p.parse_args()
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Training & Validation Loops
# ══════════════════════════════════════════════════════════════════════════════

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
        clinical_code = batch['clinical_code'].to(args.device)
        
        B = mri.size(0)
        
        # 1. Forward Pass (3 outputs: subtype, stage heads, clinical diagnosis head)
        subtype_logits, stage_logits_list, diag_logits = model(mri)
        
        # 2. Subtype Loss: Soft Target Cross-Entropy
        log_probs = F.log_softmax(subtype_logits, dim=-1)
        loss_sub = -(subtype_probs * log_probs).sum(dim=-1).mean()
        
        # 3. Stage Loss: Ordinal cumulative BCE with Pure Hard Masking
        ordinal_targets = torch.zeros(B, 48, device=args.device)
        for i in range(B):
            S = int(torch.clamp(assigned_stage[i], min=0, max=48).item())
            ordinal_targets[i, :S] = 1.0
            
        stacked_stage_logits = torch.stack(stage_logits_list, dim=1) # (B, 3, 48)
        batch_indices = torch.arange(B, device=args.device)
        assigned_logits = stacked_stage_logits[batch_indices, assigned_subtype, :] # (B, 48)
        
        loss_stage = F.binary_cross_entropy_with_logits(assigned_logits, ordinal_targets)

        # 4. Clinical Diagnosis Loss: Cross-Entropy (CN, sMCI, pMCI, AD)
        loss_diag = F.cross_entropy(diag_logits, clinical_code)
        
        # 5. Combined Multi-Task Loss & Backward Pass
        loss = args.lambda_subtype * loss_sub + args.lambda_stage * loss_stage + args.lambda_diag * loss_diag
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * B
        total_loss_sub += loss_sub.item() * B
        total_loss_stage += loss_stage.item() * B
        
    num_samples = len(loader.dataset)
    return total_loss / num_samples, total_loss_sub / num_samples, total_loss_stage / num_samples


def evaluate(model, loader, args):
    """
    Evaluates the model and computes metrics: Subtype Acc/F1, Stage MAE, QWK, Spearman Rho, and Diagnosis Acc/F1.
    """
    model.eval()
    
    total_loss = 0.0
    total_loss_sub = 0.0
    total_loss_stage = 0.0
    total_loss_diag = 0.0
    
    all_sub_preds = []
    all_sub_targets = []
    all_sub_probs_pred = []
    all_sub_probs_target = []
    all_stage_preds = []
    all_stage_targets = []
    all_diag_preds = []
    all_diag_targets = []
    
    with torch.no_grad():
        for batch in loader:
            mri = batch['mri'].to(args.device)
            subtype_probs = batch['subtype_probs'].to(args.device)
            assigned_subtype = batch['assigned_subtype'].to(args.device)
            assigned_stage = batch['assigned_stage'].to(args.device)
            clinical_code = batch['clinical_code'].to(args.device)
            
            B = mri.size(0)
            
            # Forward pass
            subtype_logits, stage_logits_list, diag_logits = model(mri)
            
            # Subtype loss
            log_probs = F.log_softmax(subtype_logits, dim=-1)
            loss_sub = -(subtype_probs * log_probs).sum(dim=-1).mean()
            
            # Stage loss target
            ordinal_targets = torch.zeros(B, 48, device=args.device)
            for i in range(B):
                S = int(torch.clamp(assigned_stage[i], min=0, max=48).item())
                ordinal_targets[i, :S] = 1.0
                
            stacked_stage_logits = torch.stack(stage_logits_list, dim=1) # (B, 3, 48)
            batch_indices = torch.arange(B, device=args.device)
            assigned_logits = stacked_stage_logits[batch_indices, assigned_subtype, :] # (B, 48)
            
            loss_stage = F.binary_cross_entropy_with_logits(assigned_logits, ordinal_targets)

            # Clinical diagnosis loss
            loss_diag = F.cross_entropy(diag_logits, clinical_code)
                
            loss = args.lambda_subtype * loss_sub + args.lambda_stage * loss_stage + args.lambda_diag * loss_diag
            
            total_loss += loss.item() * B
            total_loss_sub += loss_sub.item() * B
            total_loss_stage += loss_stage.item() * B
            total_loss_diag += loss_diag.item() * B
            
            # Predictions for metrics
            pred_sub = torch.argmax(subtype_logits, dim=-1)
            all_sub_preds.extend(pred_sub.cpu().numpy())
            all_sub_targets.extend(assigned_subtype.cpu().numpy())
            
            probs_sub_pred = torch.softmax(subtype_logits, dim=-1)
            all_sub_probs_pred.extend(probs_sub_pred.cpu().numpy())
            all_sub_probs_target.extend(subtype_probs.cpu().numpy())
            
            # Stage Prediction Reconstruction:
            probs_sub_pred = torch.softmax(subtype_logits, dim=-1)
            reconstructed_stages = torch.zeros(B, device=args.device)
            for c in range(3):
                probs_event_c = torch.sigmoid(stage_logits_list[c])
                stage_c = probs_event_c.sum(dim=-1)
                reconstructed_stages += probs_sub_pred[:, c] * stage_c
                
            all_stage_preds.extend(reconstructed_stages.cpu().numpy())
            all_stage_targets.extend(assigned_stage.cpu().numpy())

            # Diagnosis Predictions
            pred_diag = torch.argmax(diag_logits, dim=-1)
            all_diag_preds.extend(pred_diag.cpu().numpy())
            all_diag_targets.extend(clinical_code.cpu().numpy())
            
    num_samples = len(loader.dataset)
    avg_loss = total_loss / num_samples
    avg_loss_sub = total_loss_sub / num_samples
    avg_loss_stage = total_loss_stage / num_samples
    
    # Compute Metrics
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

    diag_acc = accuracy_score(all_diag_targets, all_diag_preds)
    diag_f1 = f1_score(all_diag_targets, all_diag_preds, average='macro')
    
    return avg_loss, avg_loss_sub, avg_loss_stage, acc, f1, sub_prob_mse, mae, qwk, rho, diag_acc, diag_f1


def predict_test(model, loader, args):
    """
    Evaluates the model on the test partition and collects predictions for exportation.
    """
    model.eval()
    predictions = []
    
    inv_map = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}
    with torch.no_grad():
        for batch in loader:
            mri = batch['mri'].to(args.device)
            subtype_probs = batch['subtype_probs'].to(args.device)
            assigned_subtype = batch['assigned_subtype'].to(args.device)
            assigned_stage = batch['assigned_stage'].to(args.device)
            subject_ids = batch['subject_id']
            
            subtype_logits, stage_logits_list, diag_logits = model(mri)
            
            # Predict subtype (highest logit index + 1 to match CSV 1, 2, 3)
            pred_sub = torch.argmax(subtype_logits, dim=-1).cpu().numpy() + 1
            probs_sub_pred = torch.softmax(subtype_logits, dim=-1).cpu().numpy()
            
            # Predict 4-class diagnosis
            pred_diag = torch.argmax(diag_logits, dim=-1).cpu().numpy()
            probs_diag_pred = torch.softmax(diag_logits, dim=-1).cpu().numpy()

            # Reconstruct stages
            reconstructed_stages = torch.zeros(mri.size(0), device=args.device)
            probs_sub_pred_torch = torch.softmax(subtype_logits, dim=-1)
            for c in range(3):
                probs_event_c = torch.sigmoid(stage_logits_list[c])
                stage_c = probs_event_c.sum(dim=-1)
                reconstructed_stages += probs_sub_pred_torch[:, c] * stage_c
            pred_stages = reconstructed_stages.cpu().numpy()
            
            clinical_labels = batch.get('clinical_label', ['Unknown'] * mri.size(0))
            for i in range(mri.size(0)):
                predictions.append({
                    'PTID':                  subject_ids[i],
                    'Label_True':            clinical_labels[i],
                    'Label_Pred':            inv_map.get(pred_diag[i], 'Unknown'),
                    'Assigned_Subtype_True': int(assigned_subtype[i].item()) + 1,
                    'Assigned_Subtype_Pred': int(pred_sub[i]),
                    'Assigned_Stage_True':   float(assigned_stage[i].item()),
                    'Assigned_Stage_Pred':   float(pred_stages[i]),
                    'Prob_Subtype_1_True':   float(subtype_probs[i, 0].item()),
                    'Prob_Subtype_2_True':   float(subtype_probs[i, 1].item()),
                    'Prob_Subtype_3_True':   float(subtype_probs[i, 2].item()),
                    'Prob_Subtype_1_Pred':   float(probs_sub_pred[i, 0]),
                    'Prob_Subtype_2_Pred':   float(probs_sub_pred[i, 1]),
                    'Prob_Subtype_3_Pred':   float(probs_sub_pred[i, 2]),
                    'Prob_CN_Pred':          float(probs_diag_pred[i, 0]),
                    'Prob_sMCI_Pred':        float(probs_diag_pred[i, 1]),
                    'Prob_pMCI_Pred':        float(probs_diag_pred[i, 2]),
                    'Prob_AD_Pred':          float(probs_diag_pred[i, 3])
                })
                
    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Builder Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(dataset, indices, shuffle, args):
    subset = Subset(dataset, indices)
    g = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        subset,
        batch_size     = args.batch_size,
        shuffle        = shuffle,
        num_workers    = args.num_workers,
        worker_init_fn = seed_worker,
        generator      = g,
        pin_memory     = True if 'cuda' in args.device else False
    )


def train_fold_pipeline(train_idx, val_idx, test_idx, dataset, fold_name, args):
    print(f"\n--- Training Fold/Split: {fold_name} ---")
    print(f"    Train size: {len(train_idx)} | Val size: {len(val_idx)} | Test size: {len(test_idx)}")
    
    train_loader = make_loader(dataset, train_idx, shuffle=True, args=args)
    val_loader   = make_loader(dataset, val_idx, shuffle=False, args=args)
    test_loader  = make_loader(dataset, test_idx, shuffle=False, args=args)
    
    model = SuStaInCNN().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_qwk = -1.0
    best_val_rho = 0.0
    
    output_dir = os.path.join("outputs", "runs", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Epoch Loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_sub, tr_stg = train_epoch(model, train_loader, optimizer, args)
        val_loss, val_sub, val_stg, val_acc, val_f1, val_mse, val_mae, val_qwk, val_rho, val_dacc, val_df1 = evaluate(model, val_loader, args)
        
        scheduler.step()
        
        # Log progress periodically
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"Tr Loss: {tr_loss:.4f} (Sub: {tr_sub:.3f}, Stg: {tr_stg:.3f}) | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Sub Acc: {val_acc:.3f} | Stg MAE: {val_mae:.2f} | "
                  f"Diag Acc: {val_dacc:.3f}, Diag F1: {val_df1:.3f}")
                  
        # Save Best Checkpoint
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
                'val_rho': val_rho,
                'val_diag_acc': val_dacc,
                'val_diag_f1': val_df1
            }, best_path)
            
    # Save Last Checkpoint
    latest_path = os.path.join(output_dir, f"latest_model_{fold_name}.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, latest_path)
            
    # Load Best Model for Testing
    print(f"Loading best model checkpoint for Fold {fold_name} and running inference on Test partition...")
    checkpoint = torch.load(
        os.path.join(output_dir, f"best_model_{fold_name}.pth"), 
        map_location=args.device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_sub, test_stg, test_acc, test_f1, test_mse, test_mae, test_qwk, test_rho, test_dacc, test_df1 = evaluate(model, test_loader, args)
    print(f"---> Fold {fold_name} TEST Performance: Loss: {test_loss:.4f} | Sub Acc: {test_acc:.3f} | Stg MAE: {test_mae:.2f} | Diag Acc: {test_dacc:.3f}, Diag F1: {test_df1:.3f}")
    
    test_predictions = predict_test(model, test_loader, args)
    for pred in test_predictions:
        pred['Test_Fold'] = fold_name
        
    return test_loss, test_acc, test_f1, test_mae, test_qwk, test_rho, test_dacc, test_df1, test_predictions


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_global_seed(args.seed)
    
    # Configure logs
    log_dir = os.path.join("outputs", "runs", args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    tee = save_run_config(args, os.path.join(log_dir, "train_log.txt"))
    
    print_experiment_config(args)
    
    # ── 1. Prepare Dataset ──────────────────────────────────────────────────
    if args.mock_data:
        print("Using MOCK data for local verification...")
        dataset = MockDataset(size=40)
    else:
        print(f"Loading real data from: {args.data_root}")
        dataset = SuStaInDataset(npz_root=args.data_root, csv_path=args.csv_path)
        
    all_oof_predictions = []
    
    # ── 2. Run Cross-Validation / Split ─────────────────────────────────────
    if args.kfold > 0:
        print(f"Starting {args.kfold}-Fold Stratified Cross Validation...")
        subtypes = dataset.get_labels()
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        
        test_losses = []
        test_accs = []
        test_f1s = []
        test_maes = []
        test_qwks = []
        test_rhos = []
        test_daccs = []
        test_df1s = []
        
        # Outer loop: every subject is tested exactly once
        for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), subtypes), start=1):
            
            # Split trainval into train (85%) and validation (15%) for early-stopping selection
            rng = np.random.default_rng(args.seed + fold)
            shuffled_trainval = rng.permutation(trainval_idx)
            split_point = int(0.85 * len(shuffled_trainval))
            train_idx = shuffled_trainval[:split_point]
            val_idx   = shuffled_trainval[split_point:]
            
            loss, acc, f1, mae, qwk, rho, dacc, df1, preds = train_fold_pipeline(
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
        val_size   = int(args.val_ratio * len(dataset))
        test_size  = len(dataset) - train_size - val_size
        
        trainval_ds, test_ds = random_split(
            dataset, [train_size + val_size, test_size], generator=generator
        )
        
        # Split trainval into train and validation
        train_size_split = int((args.train_ratio / (args.train_ratio + args.val_ratio)) * len(trainval_ds))
        val_size_split = len(trainval_ds) - train_size_split
        train_ds, val_ds = random_split(
            trainval_ds, [train_size_split, val_size_split], generator=generator
        )
        
        _, _, _, _, _, _, preds = train_fold_pipeline(
            train_ds.indices, val_ds.indices, test_ds.indices, dataset, "single_split", args
        )
        all_oof_predictions.extend(preds)
        
    # ── 3. Export Test Predictions to CSV ───────────────────────────────────
    pred_df = pd.DataFrame(all_oof_predictions)
    pred_csv_path = os.path.join(log_dir, f"{args.experiment_name}_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\n✓ Saved test predictions CSV to: {pred_csv_path}")
    print("The exported CSV contains both ground-truth and predictions for comparison.")
    
    # Restore stdout
    sys.stdout = tee.restore()
    print("Training process completed.")


if __name__ == '__main__':
    main()
