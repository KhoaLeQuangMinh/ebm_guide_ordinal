"""
plot_mlp_folds.py
=================
Plots SuStaIn trajectories fold-by-fold, color-coded by:
  - Left Column: True Clinical Diagnosis Label (Label)
  - Right Column: Predicted Clinical Diagnosis Label from the MLP Bridge (Label_Pred)
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
    mlp_predictions_csv = '/Users/khoale/Downloads/Alzheimer_Code/outputs/mlp_bridge_4class_predictions.csv'
    
    if not os.path.exists(mlp_predictions_csv):
        print(f"Error: MLP Predictions CSV not found at {mlp_predictions_csv}")
        return

    # Load MLP 4-class predictions CSV
    df = pd.read_csv(mlp_predictions_csv)
    print(f"Loaded {len(df)} subject predictions from MLP bridge.")

    # Filter out empty/NaN folds and sort them
    df = df.dropna(subset=['Test_Fold'])
    unique_folds = sorted(df['Test_Fold'].unique())
    n_folds = len(unique_folds)
    
    if n_folds == 0:
        print("Error: No fold information ('Test_Fold') found.")
        return

    # Set up layout grid: n_folds rows, 2 columns (True vs MLP Predicted Labels)
    fig, axes = plt.subplots(n_folds, 2, figsize=(14, 3.4 * n_folds), sharex=True, sharey=True)
    
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

        # Calculate MLP classification accuracy for this fold
        fold_acc = np.mean(df_fold['Label'] == df_fold['Label_Pred']) * 100

        # ── Column 0: Ground Truth Labels ────────────────────────────────────
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
        ax_true.set_title(f"{fold_name.upper()} — True Clinical Labels (N={len(df_fold)})")
        ax_true.set_ylabel("Subtype Trajectory")
        ax_true.set_yticks([1.0, 2.0, 3.0])
        ax_true.set_yticklabels(["Subtype 1", "Subtype 2", "Subtype 3"])

        # ── Column 1: MLP Predicted Labels ──────────────────────────────────
        ax_pred = axes[idx, 1]
        for subtype in [1.0, 2.0, 3.0]:
            ax_pred.axhline(y=subtype, color='gray', linestyle='--', alpha=0.3, zorder=1)
            
        sns.scatterplot(
            data=df_fold,
            x='Assigned_Stage_Pred',
            y='Subtype_Pred_Jittered',
            hue='Label_Pred',
            palette=palette,
            alpha=0.75,
            s=40,
            edgecolor='w',
            linewidth=0.4,
            ax=ax_pred,
            zorder=2,
            legend=(idx == 0) # Legend on first row predictions panel
        )
        ax_pred.set_title(f"{fold_name.upper()} — MLP Predicted Diagnosis Labels")
        ax_pred.set_ylabel("")

        if idx == 0:
            ax_pred.legend(title="Predicted Diagnosis", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, shadow=True)

        # Add metric tag on the predictions plot
        stats_tag = f"MLP Diagnosis Acc: {fold_acc:.1f}%"
        ax_pred.text(0.95, 0.05, stats_tag, transform=ax_pred.text_coordinates if hasattr(ax_pred, 'text_coordinates') else ax_pred.transAxes,
                     fontsize=9.5, family='monospace', horizontalalignment='right', verticalalignment='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    # Set bottom labels
    axes[-1, 0].set_xlabel("Progression Stage (0 to 48)", labelpad=8)
    axes[-1, 1].set_xlabel("Progression Stage (0 to 48)", labelpad=8)
    
    # Global limits
    plt.ylim(0.6, 3.4)
    plt.xlim(-1, 49)

    plt.suptitle("Fold-by-Fold Trajectories: True vs. MLP Predicted Diagnosis Labels", weight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.84, hspace=0.35)
    
    output_png = '/Users/khoale/Downloads/Alzheimer_Code/sustain_mlp_fold_comparison_plot.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n✓ Saved comparative per-fold MLP trajectory plot to: {output_png}")
    plt.show()

if __name__ == '__main__':
    main()
