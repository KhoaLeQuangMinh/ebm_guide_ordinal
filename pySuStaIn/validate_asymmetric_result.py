import pickle
import pandas as pd
import numpy as np
import os

# 1. Paths
data_path = "/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_sustain_prepared_asymmetric_filter_cn.csv"
pickle_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_asymmetric_filter_cn_output/pickle_files/ADNI_asym_subtype2.pickle"
output_csv_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results_filter_cn.csv"

# Load prepared dataset
df = pd.read_csv(data_path)

# List of 24 asymmetric biomarkers
regions = [
    "L_Frontal", "L_Temporal", "L_Parietal", "L_Occipital", "L_Cingulate", "L_Insula",
    "L_Hippocampus", "L_Amygdala", "L_Caudate", "L_Pallidum", "L_Putamen", "L_Accumbens",
    "R_Frontal", "R_Temporal", "R_Parietal", "R_Occipital", "R_Cingulate", "R_Insula",
    "R_Hippocampus", "R_Amygdala", "R_Caudate", "R_Pallidum", "R_Putamen", "R_Accumbens"
]
N_bio = len(regions)

# Re-create index mapping used by ZscoreSustain internally (Z=1, 2)
Z_vals = np.array([[1, 2]] * N_bio)
stage_zscore = Z_vals.T.flatten()
IX_select = np.nonzero(stage_zscore)[0]
stage_zscore = stage_zscore[IX_select]
stage_biomarker_index = np.tile(np.arange(N_bio), (len(np.unique(stage_zscore)),))
stage_biomarker_index = stage_biomarker_index[IX_select]

# Load 3-subtype SuStaIn results
with open(pickle_path, 'rb') as f:
    results = pickle.load(f)

# Extract predictions
ml_subtype = results["ml_subtype"]
ml_stage = results["ml_stage"]
prob_subtype = results["prob_subtype"]
num_subtypes = prob_subtype.shape[1]

# Assign predictions to dataframe
df["Assigned_Subtype"] = ml_subtype + 1
df["Assigned_Stage"] = ml_stage
for s in range(num_subtypes):
    df[f"Prob_Subtype_{s+1}"] = prob_subtype[:, s]

# Export single prediction CSV
df.to_csv(output_csv_path, index=False)
print("="*80)
print(f" SUCCESS! Exported staging results to single CSV:\n {output_csv_path}")
print("="*80)

# Print Summary Tables
print("\n--- Subtype Distribution across Clinical Labels ---")
print(pd.crosstab(df["Label"], df["Assigned_Subtype"]))

print("\n--- Stage Distribution across Clinical Labels ---")
pd.set_option('display.max_columns', None)
print(pd.crosstab(df["Label"], df["Assigned_Stage"]))