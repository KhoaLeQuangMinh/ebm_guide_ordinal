"""
plot_folds.py
=============
Plots comparative SuStaIn trajectories (Subtype vs Stage) side-by-side (True vs Pred)
individually for each fold, color-coded by clinical diagnosis label (CN, sMCI, pMCI, AD).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional visualization styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def main():
    predictions_csv = '/Users/khoale/Downloads/Alzheimer_Code/sustain_guided_predictions.csv'
    reference_csv = '/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results.csv'
    merged_mri_csv = '/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_ucsf_merged.csv'
    
    if not os.path.exists(predictions_csv):
        print(f"Error: Predictions CSV not found at {predictions_csv}")
        return

    # Load predictions CSV
    df_pred = pd.read_csv(predictions_csv)
    print(f"Loaded {len(df_pred)} predictions.")

    # Find clinical Labels (CN, sMCI, pMCI, AD) to merge on PTID
    df_labels = None
    for path in [reference_csv, merged_mri_csv]:
        if os.path.exists(path):
            df_ref = pd.read_csv(path)
            if 'PTID' in df_ref.columns and 'Label' in df_ref.columns:
                df_labels = df_ref[['PTID', 'Label']].drop_duplicates(subset=['PTID'])
                print(f"Loaded clinical labels from {os.path.basename(path)}")
                break
                
    if df_labels is not None:
        df = df_pred.merge(df_labels, on='PTID', how='left')
    else:
        print("Warning: Clinical labels ('Label') not found. Plotting without diagnosis coloring.")
        df = df_pred.copy()
        df['Label'] = 'Unknown'

    # Filter out empty/NaN folds and sort them
    df = df.dropna(subset=['Test_Fold'])
    unique_folds = sorted(df['Test_Fold'].unique())
    n_folds = len(unique_folds)
    
    if n_folds == 0:
        print("Error: No fold information ('Test_Fold') found in predictions CSV.")
        return

    # Set up layout grid: n_folds rows, 2 columns (True vs Pred)
    fig, axes = plt.subplots(n_folds, 2, figsize=(14, 3.4 * n_folds), sharex=True, sharey=True)
    
    # Handle single fold case
    if n_folds == 1:
        axes = np.array([axes])

    # Distinct color palette matching clinical conventions
    palette = {
        'CN': '#2ca02c',    # Green
        'sMCI': '#1f77b4',  # Blue
        'pMCI': '#ff7f0e',  # Orange
        'AD': '#d62728',     # Red
        'Unknown': '#7f7f7f'
    }

    rng = np.random.default_rng(42)

    # Plot each fold row-by-row
    for idx, fold_name in enumerate(unique_folds):
        df_fold = df[df['Test_Fold'] == fold_name].copy()
        
        # Add local jittering for this fold
        jitter = rng.normal(loc=0.0, scale=0.08, size=len(df_fold))
        df_fold['Subtype_True_Jittered'] = df_fold['Assigned_Subtype_True'] + jitter
        df_fold['Subtype_Pred_Jittered'] = df_fold['Assigned_Subtype_Pred'] + jitter

        # Subtype accuracy & Stage MAE for this fold
        fold_acc = np.mean(df_fold['Assigned_Subtype_True'] == df_fold['Assigned_Subtype_Pred']) * 100
        fold_mae = np.mean(np.abs(df_fold['Assigned_Stage_True'] - df_fold['Assigned_Stage_Pred']))

        # ── Column 0: Ground Truth ───────────────────────────────────────────
        ax_true = axes[idx, 0]
        for subtype in [1.0, 2.0, 3.0]:
            ax_true.axhline(y=subtype, color='gray', linestyle='--', alpha=0.3, zorder=1)
            
        sns.scatterplot(
            data=df_fold,
            x='Assigned_Stage_True',
            y='Subtype_True_Jittered',
            hue='Label',
            palette=palette,
            alpha=0.75,
            s=40,
            edgecolor='w',
            linewidth=0.4,
            ax=ax_true,
            zorder=2,
            legend=False
        )
        ax_true.set_title(f"{fold_name.upper()} — Ground Truth (N={len(df_fold)})")
        ax_true.set_ylabel("Subtype Trajectory")
        ax_true.set_yticks([1.0, 2.0, 3.0])
        ax_true.set_yticklabels(["Subtype 1", "Subtype 2", "Subtype 3"])

        # ── Column 1: Model Predictions ──────────────────────────────────────
        ax_pred = axes[idx, 1]
        for subtype in [1.0, 2.0, 3.0]:
            ax_pred.axhline(y=subtype, color='gray', linestyle='--', alpha=0.3, zorder=1)
            
        sns.scatterplot(
            data=df_fold,
            x='Assigned_Stage_Pred',
            y='Subtype_Pred_Jittered',
            hue='Label',
            palette=palette,
            alpha=0.75,
            s=40,
            edgecolor='w',
            linewidth=0.4,
            ax=ax_pred,
            zorder=2,
            legend=(idx == 0) # Only draw legend on first row predictions panel
        )
        ax_pred.set_title(f"{fold_name.upper()} — CNN Predictions")
        ax_pred.set_ylabel("")

        if idx == 0:
            ax_pred.legend(title="Diagnosis Group", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, shadow=True)

        # Add metric tag on the predictions plot
        stats_tag = f"Subtype Acc: {fold_acc:.1f}%\nStage MAE: {fold_mae:.2f}"
        ax_pred.text(0.95, 0.05, stats_tag, transform=ax_pred.transAxes, fontsize=9, family='monospace',
                     horizontalalignment='right', verticalalignment='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    # Set bottom labels
    axes[-1, 0].set_xlabel("Progression Stage (0 to 48)", labelpad=8)
    axes[-1, 1].set_xlabel("Progression Stage (0 to 48)", labelpad=8)
    
    # Global limits
    plt.ylim(0.6, 3.4)
    plt.xlim(-1, 49)

    plt.suptitle("Comparative Staging Trajectories per Fold (True vs. Predicted)", weight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.85, hspace=0.35)
    
    output_png = '/Users/khoale/Downloads/Alzheimer_Code/sustain_fold_comparison_plot.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n✓ Saved comparative per-fold trajectories with diagnosis labels to: {output_png}")
    plt.show()

if __name__ == '__main__':
    main()
