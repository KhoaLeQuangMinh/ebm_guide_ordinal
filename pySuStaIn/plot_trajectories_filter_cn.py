import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Path to predictions CSV
csv_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results_filter_cn.csv"
output_plot_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_trajectories_filter_cn.png"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Prediction file not found: {csv_path}. Please run validate_asymmetric_result.py first!")

df = pd.read_csv(csv_path)

# Set up styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Define diagnosis colors and order
LABEL_ORDER = ["CN", "sMCI", "pMCI", "AD"]
PALETTE = {
    "CN": "#2ca02c",    # Green
    "sMCI": "#1f77b4",  # Blue
    "pMCI": "#ff7f0e",  # Orange
    "AD": "#d62728"     # Red
}

# Filter to valid labels if needed
df_plot = df[df["Label"].isin(LABEL_ORDER)].copy()

# Add small vertical jitter to Subtype for visual clarity (prevents points from overlapping)
np.random.seed(42)
df_plot["Subtype_Jittered"] = df_plot["Assigned_Subtype"] + np.random.uniform(-0.15, 0.15, size=len(df_plot))

# Setup figure
fig, ax = plt.subplots(figsize=(14, 7))

# Draw horizontal lines for each subtype trajectory
subtypes = sorted(df_plot["Assigned_Subtype"].unique())
for sub in subtypes:
    ax.axhline(sub, color="lightgray", linestyle="--", linewidth=1, zorder=1)

# Scatter plot of trajectory stages colored by Diagnosis
sns.scatterplot(
    data=df_plot,
    x="Assigned_Stage",
    y="Subtype_Jittered",
    hue="Label",
    hue_order=LABEL_ORDER,
    palette=PALETTE,
    alpha=0.85,
    s=80,
    edgecolor="white",
    linewidth=0.6,
    ax=ax,
    zorder=2
)

max_stage = int(df_plot["Assigned_Stage"].max())

ax.set_title("SuStaIn Subtype Trajectories & Clinical Staging (Clean CN Filtered Baseline)", fontsize=14, weight="bold", pad=15)
ax.set_xlabel("Assigned Progression Stage", fontsize=12, labelpad=10)
ax.set_ylabel("Assigned Subtype Trajectory", fontsize=12, labelpad=10)
ax.set_yticks(subtypes)
ax.set_yticklabels([f"Subtype {int(s)}" for s in subtypes])
ax.set_ylim(min(subtypes) - 0.5, max(subtypes) + 0.5)
ax.set_xlim(-1, max_stage + 2)
ax.legend(title="Clinical Diagnosis", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, shadow=True)

# Build text summary box of subtype breakdown on the right side
stats_text = "Subtype Distribution:\n"
for sub in subtypes:
    sub_df = df_plot[df_plot["Assigned_Subtype"] == sub]
    total_sub = len(sub_df)
    stats_text += f"\nSubtype {int(sub)} (N={total_sub}):\n"
    if total_sub > 0:
        counts = sub_df["Label"].value_counts()
        for lbl in LABEL_ORDER:
            if lbl in counts:
                cnt = counts[lbl]
                pct = (cnt / total_sub) * 100
                stats_text += f"  - {lbl}: {cnt} ({pct:.1f}%)\n"

fig.text(
    0.77, 0.15, stats_text, fontsize=9.5, family="monospace", verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
)

plt.subplots_adjust(right=0.74, left=0.08, top=0.90, bottom=0.12)
plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
print("="*80)
print(f" SUCCESS! Trajectory plot saved to:\n {output_plot_path}")
print("="*80)
