import pickle
import pandas as pd
import numpy as np
import os

# 1. Paths
data_path = "/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_sustain_prepared_asymmetric.csv"
pickle_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_asymmetric_output/pickle_files/ADNI_asym_subtype2.pickle"
output_csv_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results.csv"

# Load the prepared dataset to match labels
df = pd.read_csv(data_path)

with open(pickle_path, 'rb') as f:
    results = pickle.load(f)

# Extract variables from the 3-subtype model
ml_subtype = results["ml_subtype"]  # Subtype (0, 1, 2)
ml_stage = results["ml_stage"]      # Stage (0 to 48)
prob_subtype = results["prob_subtype"]  # Probabilities for each subtype

# Add results to our dataframe
df["Assigned_Subtype"] = ml_subtype + 1
df["Assigned_Stage"] = ml_stage
df["Prob_Subtype_1"] = prob_subtype[:, 0]
df["Prob_Subtype_2"] = prob_subtype[:, 1]
df["Prob_Subtype_3"] = prob_subtype[:, 2]

# Save the detailed patient mapping to a spreadsheet
df.to_csv(output_csv_path, index=False)
print(f"Exported detailed subject staging to: {output_csv_path}")

# 2. Group the 48 stages into clinical phases for readability
def get_stage_phase(stage):
    if stage == 0:
        return "Stage 0 (No Atrophy)"
    elif 1 <= stage <= 16:
        return "Stages 1-16 (Early Atrophy)"
    elif 17 <= stage <= 32:
        return "Stages 17-32 (Moderate Atrophy)"
    else:
        return "Stages 33-48 (Severe Atrophy)"

df["Disease_Phase"] = df["Assigned_Stage"].apply(get_stage_phase)

# 3. Print the staging profile for each subtype
print("\n" + "="*80)
print(" PATIENT DISTRIBUTION ACROSS THE 3 SUBTYPES & STAGES ".center(80, "="))
print("="*80)

phase_order = [
    "Stage 0 (No Atrophy)",
    "Stages 1-16 (Early Atrophy)",
    "Stages 17-32 (Moderate Atrophy)",
    "Stages 33-48 (Severe Atrophy)"
]

for s in [1, 2, 3]:
    subtype_df = df[df["Assigned_Subtype"] == s]
    
    # Define subtype names based on their biological sequences
    subtype_names = {
        1: "Subtype 1: Subcortical-First",
        2: "Subtype 2: Typical AD (Limbic-First)",
        3: "Subtype 3: Asymmetric Left-Hemisphere Dominant"
    }
    
    print(f"\n>>> {subtype_names[s]} (n = {len(subtype_df)})")
    
    # Generate the pivot table
    pivot = pd.crosstab(subtype_df["Label"], subtype_df["Disease_Phase"])
    
    # Reorder columns logically
    existing_phases = [p for p in phase_order if p in pivot.columns]
    pivot = pivot[existing_phases]
    
    print(pivot)
    print("-" * 80)