import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# 1. File Paths
csv_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results_filter_cn.csv"
pickle_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_asymmetric_filter_cn_output/pickle_files/ADNI_asym_subtype2.pickle"
output_plot_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_trajectories_with_events_filter_cn.png"
output_matrix_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_event_order_comparison.png"
output_seq_csv = "/Users/khoale/Downloads/Alzheimer_Code/sustain_subtype_event_sequences.csv"
artifact_dir = "/Users/khoale/.gemini/antigravity/brain/3b8e56ac-7bd4-4903-9cac-2d2583776938"

# 2. Load data
df = pd.read_csv(csv_path)

with open(pickle_path, 'rb') as f:
    results = pickle.load(f)

regions = [
    "L_Frontal", "L_Temporal", "L_Parietal", "L_Occipital", "L_Cingulate", "L_Insula",
    "L_Hippocampus", "L_Amygdala", "L_Caudate", "L_Pallidum", "L_Putamen", "L_Accumbens",
    "R_Frontal", "R_Temporal", "R_Parietal", "R_Occipital", "R_Cingulate", "R_Insula",
    "R_Hippocampus", "R_Amygdala", "R_Caudate", "R_Pallidum", "R_Putamen", "R_Accumbens"
]

# Event names: z=1 (mild atrophy) and z=2 (moderate/severe atrophy)
event_names = [f"{r} (z=1)" for r in regions] + [f"{r} (z=2)" for r in regions]
ml_seq = results["ml_sequence_EM"] # shape (3, 48)

# Export event sequences to CSV for easy reference
seq_records = []
for s_idx in range(3):
    seq = ml_seq[s_idx].astype(int)
    for stage_idx, ev_idx in enumerate(seq, 1):
        seq_records.append({
            "Subtype": s_idx + 1,
            "Stage": stage_idx,
            "Event_Index": ev_idx,
            "Event_Name": event_names[ev_idx],
            "Region": regions[ev_idx % 24],
            "Z_Score": 1 if ev_idx < 24 else 2
        })
df_seq = pd.DataFrame(seq_records)
df_seq.to_csv(output_seq_csv, index=False)

# Styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

LABEL_ORDER = ["CN", "sMCI", "pMCI", "AD"]
PALETTE = {
    "CN": "#2ca02c",    # Green
    "sMCI": "#1f77b4",  # Blue
    "pMCI": "#ff7f0e",  # Orange
    "AD": "#d62728"     # Red
}

subtypes = [1, 2, 3]
subtype_titles = {
    1: "Subtype 1: Cortical / Temporo-Frontal Onset",
    2: "Subtype 2: Medial Temporal Lobe (Hippocampal/Amygdala) Onset",
    3: "Subtype 3: Subcortical (Striatal/Basal Ganglia) Onset"
}

# -------------------------------------------------------------
# PLOT 1: Subject Staging Trajectories with Actual Event Names
# -------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=False)
fig.subplots_adjust(hspace=0.65)

np.random.seed(42)

for idx, sub in enumerate(subtypes):
    ax = axes[idx]
    sub_seq = ml_seq[sub - 1].astype(int)
    events_in_order = ["Stage 0: Normal"] + [f"S{stage_idx:02d}: {event_names[i]}" for stage_idx, i in enumerate(sub_seq, 1)]
    
    # Filter subjects for this subtype
    df_sub = df[df["Assigned_Subtype"] == sub].copy()
    
    # Add vertical jitter to y-axis (y=1 for subjects)
    df_sub["Y_Jitter"] = 1.0 + np.random.uniform(-0.22, 0.22, size=len(df_sub))
    
    # Draw horizontal baseline track
    ax.axhline(1.0, color="#888888", linestyle="-", linewidth=1.5, zorder=1)
    
    # Scatter plot of subjects
    sns.scatterplot(
        data=df_sub,
        x="Assigned_Stage",
        y="Y_Jitter",
        hue="Label",
        hue_order=LABEL_ORDER,
        palette=PALETTE,
        alpha=0.85,
        s=65,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
        zorder=3
    )
    
    # Formatting
    ax.set_title(f"{subtype_titles[sub]} (N={len(df_sub)})", fontsize=12, weight="bold", loc="left", color="#111111")
    ax.set_yticks([1.0])
    ax.set_yticklabels(["Subjects"], fontsize=10, weight="bold")
    ax.set_ylim(0.65, 1.35)
    ax.set_xlim(-0.8, 48.8)
    
    # Set x-ticks to show actual stage events
    ax.set_xticks(range(49))
    ax.set_xticklabels(events_in_order, rotation=90, ha="center", fontsize=7.5)
    
    # Annotate stages with grid lines
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.grid(False, axis="y")
    
    # Add legend to top panel only
    if idx == 0:
        ax.legend(title="Clinical Diagnosis", loc="upper right", frameon=True, facecolor="white", edgecolor="gray")
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

fig.suptitle("SuStaIn Disease Progression Trajectories with Specific Brain Region Events (Clean CN Filtered Baseline)", fontsize=14, weight="bold", y=0.99)
plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------
# PLOT 2: Event Progression Stage Matrix across Subtypes
# -------------------------------------------------------------
# Matrix where rows = 48 events, cols = Subtype 1, Subtype 2, Subtype 3, cell value = Stage (1..48)
matrix_data = np.zeros((48, 3), dtype=int)
for s_idx in range(3):
    seq = ml_seq[s_idx].astype(int)
    for stage_idx, ev_idx in enumerate(seq, 1):
        matrix_data[ev_idx, s_idx] = stage_idx

df_matrix = pd.DataFrame(matrix_data, index=event_names, columns=["Subtype 1 (Cortical)", "Subtype 2 (MTL)", "Subtype 3 (Subcortical)"])

# Sort rows by early progression in Subtype 1 then Subtype 2
df_matrix_sorted = df_matrix.sort_values(by=["Subtype 1 (Cortical)"])

fig2, ax2 = plt.subplots(figsize=(10, 16))
sns.heatmap(df_matrix_sorted, annot=True, fmt="d", cmap="YlGnBu_r", cbar_kws={'label': 'Progression Stage (1 = First Event, 48 = Last Event)'}, ax=ax2, linewidths=0.5)
ax2.set_title("Biomarker Event Stage Comparison Across SuStaIn Subtypes", fontsize=13, weight="bold", pad=15)
ax2.set_ylabel("Biomarker Atrophy Event (z=1: Mild, z=2: Moderate/Severe)", fontsize=11, weight="bold")
plt.savefig(output_matrix_path, dpi=300, bbox_inches="tight")
plt.close(fig2)

# Copy to artifacts directory
shutil.copy(output_plot_path, os.path.join(artifact_dir, "sustain_trajectories_with_events_filter_cn.png"))
shutil.copy(output_matrix_path, os.path.join(artifact_dir, "sustain_event_order_comparison.png"))

print("="*80)
print(f" SUCCESS!")
print(f" 1. Trajectory plot with events: {output_plot_path}")
print(f" 2. Event matrix comparison plot: {output_matrix_path}")
print(f" 3. Event sequence CSV: {output_seq_csv}")
print("="*80)
