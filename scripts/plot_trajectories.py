"""
plot_trajectories.py
====================
Visualizes the distribution of subjects (colored by diagnosis) along their
assigned SuStaIn subtypes and progression stages.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional presentation
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def main():
    csv_path = '/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results_as.csv'
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} subjects.")
    
    # Check unique labels and subtypes
    print("\n--- Diagnosis Label Counts ---")
    print(df['Label'].value_counts())
    
    print("\n--- Subtype Assignment Counts ---")
    print(df['Assigned_Subtype'].value_counts())
    
    # ── 1. Create the Trajectory Plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define a clean color palette matching clinical conventions
    # CN: Normal Control (Green)
    # sMCI: Stable MCI (Blue)
    # pMCI: Progressive MCI (Orange)
    # AD: Alzheimer's Disease (Red)
    palette = {
        'CN': '#2ca02c',    # Green
        'sMCI': '#1f77b4',  # Blue
        'pMCI': '#ff7f0e',  # Orange
        'AD': '#d62728'     # Red
    }
    
    # Fallback to default colors if other labels are present
    unique_labels = df['Label'].unique()
    for lbl in unique_labels:
        if lbl not in palette:
            palette[lbl] = '#7f7f7f' # Grey fallback
            
    # Add small random jitter to Subtype values (y-axis) to prevent overlap of scatter points
    rng = np.random.default_rng(42)
    df['Subtype_Jittered'] = df['Assigned_Subtype'] + rng.normal(loc=0.0, scale=0.08, size=len(df))
    
    # Plot guide lines for the 3 subtypes
    for subtype in [1.0, 2.0, 3.0]:
        ax.axhline(y=subtype, color='gray', linestyle='--', alpha=0.3, zorder=1)
        
    # Scatter plot
    sns.scatterplot(
        data=df,
        x='Assigned_Stage',
        y='Subtype_Jittered',
        hue='Label',
        palette=palette,
        alpha=0.8,
        s=70,
        edgecolor='w',
        linewidth=0.5,
        ax=ax,
        zorder=2
    )
    
    # Title and Labels
    ax.set_title("Distribution of Subjects Along SuStaIn Subtypes and Progression Stages", pad=20, weight='bold')
    ax.set_xlabel("Assigned Progression Stage (0 to 48)", labelpad=10)
    ax.set_ylabel("Assigned Subtype Trajectory", labelpad=10)
    
    # Format Y-axis to display clean subtype ticks
    ax.set_yticks([1.0, 2.0, 3.0])
    ax.set_yticklabels(["Subtype 1", "Subtype 2", "Subtype 3"])
    ax.set_ylim(0.6, 3.4)
    ax.set_xlim(-1, 49)
    
    # Customize legend (moved to the right side)
    ax.legend(title="Diagnosis Group", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, shadow=True)
    
    # ── 2. Add statistics annotation ─────────────────────────────────────────
    # Let's count label distribution per subtype
    stats_text = "Subtype Distributions:\n"
    for sub in [1.0, 2.0, 3.0]:
        sub_df = df[df['Assigned_Subtype'] == sub]
        total_sub = len(sub_df)
        stats_text += f"\nSubtype {int(sub)} (N={total_sub}):\n"
        for label, count in sub_df['Label'].value_counts().items():
            pct = (count / total_sub) * 100
            stats_text += f"  - {label}: {count} ({pct:.1f}%)\n"
            
    # Place text box on the right panel, below the legend
    fig.text(0.77, 0.15, stats_text, fontsize=10, family='monospace', verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))
    
    # Adjust margins to accommodate the legend and text box on the right
    plt.subplots_adjust(right=0.75, left=0.08, top=0.92, bottom=0.1)
    
    output_png = '/Users/khoale/Downloads/Alzheimer_Code/sustain_trajectories_plot_as.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n✓ Saved plot successfully to: {output_png}")
    plt.show()

if __name__ == '__main__':
    main()
