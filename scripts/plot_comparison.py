"""
plot_comparison.py
==================
Plots a clean, faceted 3x2 grid comparing Ground Truth (True) vs. Model Predictions (Pred)
for each SuStaIn subtype. This layout separates the subtypes to show distributions clearly
without vertical overlap.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional visualization styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16
})

def main():
    predictions_csv = '/Users/khoale/Downloads/Alzheimer_Code/sustain_guided_predictions.csv'
    reference_csv = '/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results.csv'
    merged_mri_csv = '/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_ucsf_merged.csv'
    
    if not os.path.exists(predictions_csv):
        print(f"Error: Predictions CSV not found at {predictions_csv}")
        return

    # Load predictions
    df_pred = pd.read_csv(predictions_csv)

    # Try to find clinical Labels (CN, sMCI, pMCI, AD) to merge on PTID
    df_labels = None
    for path in [reference_csv, merged_mri_csv]:
        if os.path.exists(path):
            df_ref = pd.read_csv(path)
            if 'PTID' in df_ref.columns and 'Label' in df_ref.columns:
                df_labels = df_ref[['PTID', 'Label']].drop_duplicates(subset=['PTID'])
                break
                
    if df_labels is not None:
        df = df_pred.merge(df_labels, on='PTID', how='left')
    else:
        df = df_pred.copy()
        df['Label'] = 'Unknown'

    # ── 1. Set up a 3x2 grid of subplots ─────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    
    palette = {
        'CN': '#2ca02c',    # Green
        'sMCI': '#1f77b4',  # Blue
        'pMCI': '#ff7f0e',  # Orange
        'AD': '#d62728',     # Red
        'Unknown': '#7f7f7f'
    }

    rng = np.random.default_rng(42)

    # Loop through each subtype (Rows 1 to 3)
    for row_idx, subtype in enumerate([1.0, 2.0, 3.0]):
        # Filter datasets for the specific subtype
        df_true_sub = df[df['Assigned_Subtype_True'] == subtype].copy()
        df_pred_sub = df[df['Assigned_Subtype_Pred'] == subtype].copy()

        # Add vertical jitter centered at 0.0 for spacing
        df_true_sub['Jitter'] = rng.uniform(-0.25, 0.25, size=len(df_true_sub))
        df_pred_sub['Jitter'] = rng.uniform(-0.25, 0.25, size=len(df_pred_sub))

        # Col 0: Ground Truth
        ax_true = axes[row_idx, 0]
        sns.scatterplot(
            data=df_true_sub,
            x='Assigned_Stage_True',
            y='Jitter',
            hue='Label',
            palette=palette,
            alpha=0.6,
            s=45,
            edgecolor='w',
            linewidth=0.4,
            ax=ax_true,
            legend=False
        )
        ax_true.set_title(f"Ground Truth — Subtype {int(subtype)} (N={len(df_true_sub)})")
        ax_true.set_ylabel("Density Jitter")
        ax_true.set_ylim(-0.5, 0.5)
        ax_true.set_yticks([]) # Clear y-ticks as they represent arbitrary jitter
        
        # Col 1: Model Predictions
        ax_pred = axes[row_idx, 1]
        sns.scatterplot(
            data=df_pred_sub,
            x='Assigned_Stage_Pred',
            y='Jitter',
            hue='Label',
            palette=palette,
            alpha=0.6,
            s=45,
            edgecolor='w',
            linewidth=0.4,
            ax=ax_pred,
            legend=(row_idx == 0) # Only draw legend on first row
        )
        ax_pred.set_title(f"CNN Predictions — Subtype {int(subtype)} (N={len(df_pred_sub)})")
        ax_pred.set_ylabel("")
        
        if row_idx == 0:
            ax_pred.legend(title="Diagnosis Group", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, shadow=True)

    # Set bottom labels
    axes[2, 0].set_xlabel("Progression Stage (0 to 48)", labelpad=10)
    axes[2, 1].set_xlabel("Progression Stage (0 to 48)", labelpad=10)
    
    # ── 2. Add Stats overlay ─────────────────────────────────────────────────
    sub_acc = np.mean(df['Assigned_Subtype_True'] == df['Assigned_Subtype_Pred']) * 100
    stage_mae = np.mean(np.abs(df['Assigned_Stage_True'] - df['Assigned_Stage_Pred']))
    correlation = df['Assigned_Stage_True'].corr(df['Assigned_Stage_Pred'], method='spearman')
    
    stats_text = (
        f"Alignment Summary:\n"
        f"Subtype Accuracy: {sub_acc:.1f}%\n"
        f"Stage MAE:        {stage_mae:.2f} events\n"
        f"Spearman Rho:     {correlation:.3f}"
    )
    fig.text(0.81, 0.45, stats_text, fontsize=11, family='monospace', verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.suptitle("Faceted Patient Progression Analysis: True vs. Predicted Trajectories", weight='bold', y=0.98)
    plt.subplots_adjust(left=0.06, right=0.80, top=0.91, bottom=0.08, hspace=0.35)
    
    output_png = '/Users/khoale/Downloads/Alzheimer_Code/sustain_comparison_plot.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n✓ Saved clean faceted plot successfully to: {output_png}")
    plt.show()

if __name__ == '__main__':
    main()
