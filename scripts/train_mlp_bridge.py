"""
train_mlp_bridge.py
===================
Method 1 (Solution A): Two-Stage MLP Bridge with Class-Weighted Cross-Entropy Loss.
Forces the model to pay equal attention to intermediate classes (sMCI, pMCI)
by penalizing misclassifications inversely proportional to class frequencies.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── PyTorch MLP Architecture ──────────────────────────────────────────────────
class ClassWeightedMLP(nn.Module):
    def __init__(self, input_dim=4, num_classes=4):
        super(ClassWeightedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def main():
    predictions_csv = '/Users/khoale/Downloads/Alzheimer_Code/sustain_guided_predictions.csv'
    reference_csv = '/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results.csv'
    
    if not os.path.exists(predictions_csv):
        print(f"Error: Predictions CSV not found at {predictions_csv}")
        return

    # 1. Load predictions CSV
    df = pd.read_csv(predictions_csv)
    print(f"Loaded {len(df)} subject predictions.")

    # If Label column is not present, merge from reference CSV
    if 'Label' not in df.columns and os.path.exists(reference_csv):
        df_ref = pd.read_csv(reference_csv)[['PTID', 'Label']].drop_duplicates(subset=['PTID'])
        df = df.merge(df_ref, on='PTID', how='inner')
        print("Merged clinical diagnosis labels ('Label') from reference CSV.")

    # Map clinical string labels to clean integer codes
    label_mapping = {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'AD': 3}
    inverse_mapping = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}
    df['Label_Code'] = df['Label'].map(label_mapping)
    
    df = df.dropna(subset=['Label_Code', 'Test_Fold'])
    df['Label_Code'] = df['Label_Code'].astype(int)

    unique_folds = sorted(df['Test_Fold'].unique())
    print(f"Starting 5-Fold Class-Weighted MLP Training on folds: {unique_folds}\n")

    oof_predictions = []

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 3. Perform Fold-by-Fold Training & Evaluation
    for fold_name in unique_folds:
        train_df = df[df['Test_Fold'] != fold_name].copy()
        test_df  = df[df['Test_Fold'] == fold_name].copy()

        # Input Features (4 numbers)
        # Train on GROUND TRUTH SuStaIn features
        X_train_raw = train_df[['Prob_Subtype_1_True', 'Prob_Subtype_2_True', 'Prob_Subtype_3_True', 'Assigned_Stage_True']].values
        y_train     = train_df['Label_Code'].values

        # Test on CNN PREDICTED SuStaIn features
        X_test_raw  = test_df[['Prob_Subtype_1_Pred', 'Prob_Subtype_2_Pred', 'Prob_Subtype_3_Pred', 'Assigned_Stage_Pred']].values
        y_test      = test_df['Label_Code'].values

        # Feature Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        # Compute Balanced Class Weights for Cross-Entropy Loss
        classes = np.array([0, 1, 2, 3])
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.FloatTensor(weights)

        # Convert to Tensors
        X_tr_t = torch.FloatTensor(X_train)
        y_tr_t = torch.LongTensor(y_train)
        X_te_t = torch.FloatTensor(X_test)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Model, Loss, and Optimizer
        model = ClassWeightedMLP(input_dim=4, num_classes=4)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

        # Train Loop (150 Epochs)
        model.train()
        for epoch in range(150):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation on Test Fold
        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            probs  = torch.softmax(logits, dim=-1).numpy()
            y_pred_code = torch.argmax(logits, dim=-1).numpy()

        acc = accuracy_score(y_test, y_pred_code)
        f1  = f1_score(y_test, y_pred_code, average='macro')

        print(f"--- {fold_name.upper()} (Class-Weighted) ---")
        print(f"    Test Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.3f}")

        # Store test predictions
        test_df['Label_Pred_Code'] = y_pred_code
        test_df['Label_Pred']      = test_df['Label_Pred_Code'].map(inverse_mapping)
        
        for c_idx, c_name in inverse_mapping.items():
            test_df[f'Prob_{c_name}'] = probs[:, c_idx]

        oof_predictions.append(test_df)

    # 4. Compile Out-Of-Fold (OOF) Overall Summary
    df_oof = pd.concat(oof_predictions, axis=0).reset_index(drop=True)
    
    total_acc = accuracy_score(df_oof['Label_Code'], df_oof['Label_Pred_Code'])
    total_f1  = f1_score(df_oof['Label_Code'], df_oof['Label_Pred_Code'], average='macro')
    
    print("\n" + "="*70)
    print(" CLASS-WEIGHTED OOF 4-CLASS DIAGNOSTIC SUMMARY ".center(70, "="))
    print("="*70)
    print(f"Overall OOF Accuracy: {total_acc * 100:.2f}%")
    print(f"Overall OOF Macro F1: {total_f1:.4f}\n")

    print("Classification Report:")
    target_names = ['CN', 'sMCI', 'pMCI', 'AD']
    print(classification_report(df_oof['Label_Code'], df_oof['Label_Pred_Code'], target_names=target_names, digits=3))

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    cm = confusion_matrix(df_oof['Label_Code'], df_oof['Label_Pred_Code'])
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in target_names], columns=[f"Pred_{c}" for c in target_names])
    print(cm_df)
    print("="*70)

    # 5. Export Predictions CSV
    output_dir = "/Users/khoale/Downloads/Alzheimer_Code/outputs"
    os.makedirs(output_dir, exist_ok=True)
    export_path = os.path.join(output_dir, "mlp_bridge_4class_predictions.csv")
    
    export_cols = [
        'PTID', 'Label', 'Label_Pred', 'Test_Fold',
        'Assigned_Subtype_True', 'Assigned_Subtype_Pred',
        'Assigned_Stage_True', 'Assigned_Stage_Pred',
        'Prob_CN', 'Prob_sMCI', 'Prob_pMCI', 'Prob_AD'
    ]
    df_oof[export_cols].to_csv(export_path, index=False)
    print(f"\n✓ Exported class-weighted predicted labels CSV to: {export_path}")

if __name__ == '__main__':
    main()
