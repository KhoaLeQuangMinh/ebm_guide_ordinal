import pickle
import pandas as pd
import numpy as np
import os

# 1. Paths
data_path = "/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_sustain_prepared_asymmetric.csv"
pickle_folder = "/Users/khoale/Downloads/Alzheimer_Code/sustain_asymmetric_output/pickle_files"

# Load the prepared dataset to match labels
df = pd.read_csv(data_path)

# List of the 24 asymmetric biomarkers (must match the order used during training)
regions = [
    "L_Frontal", "L_Temporal", "L_Parietal", "L_Occipital", "L_Cingulate", "L_Insula",
    "L_Hippocampus", "L_Amygdala", "L_Caudate", "L_Pallidum", "L_Putamen", "L_Accumbens",
    "R_Frontal", "R_Temporal", "R_Parietal", "R_Occipital", "R_Cingulate", "R_Insula",
    "R_Hippocampus", "R_Amygdala", "R_Caudate", "R_Pallidum", "R_Putamen", "R_Accumbens"
]
N_bio = len(regions)

# Re-create the index mapping used by ZscoreSustain internally (2 thresholds now: Z=1, 2)
Z_vals = np.array([[1, 2]] * N_bio)
stage_zscore = Z_vals.T.flatten()
IX_select = np.nonzero(stage_zscore)[0]
stage_zscore = stage_zscore[IX_select]
stage_biomarker_index = np.tile(np.arange(N_bio), (len(np.unique(stage_zscore)),))
stage_biomarker_index = stage_biomarker_index[IX_select]

def analyze_pickle(pickle_name, num_subtypes):
    pickle_path = os.path.join(pickle_folder, pickle_name)
    if not os.path.exists(pickle_path):
        print(f"File not found: {pickle_path}")
        return
        
    print("\n" + "="*80)
    print(f" ANALYZING ASYMMETRIC MODEL: {num_subtypes} SUBTYPE(S) ".center(80, "="))
    print("="*80)
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
        
    ml_subtype = results["ml_subtype"]
    ml_stage = results["ml_stage"]
    ml_f_EM = results["ml_f_EM"]
    ml_sequence_EM = results["ml_sequence_EM"]
    prob_subtype = results.get("prob_subtype", None)
    
    # 1. Print subtype proportions (fractions)
    print("\n--- Estimated Subtype Proportions ---")
    for s in range(num_subtypes):
        val = ml_f_EM[s] if hasattr(ml_f_EM, "__len__") else ml_f_EM
        print(f"  Subtype {s+1}: {val*100:.1f}% of the cohort")
        
    # 2. Stage distribution across clinical labels
    temp_df = df.copy()
    temp_df["Subtype"] = ml_subtype + 1
    temp_df["Stage"] = ml_stage
    
    print("\n--- Clinical Label vs Stage Distribution ---")
    stage_table = pd.crosstab(temp_df["Label"], temp_df["Stage"])
    pd.set_option('display.max_columns', None)
    print(stage_table)
    
    # 3. Subtype distribution across clinical labels
    if num_subtypes > 1:
        print("\n--- Clinical Label vs Subtype Distribution ---")
        subtype_table = pd.crosstab(temp_df["Label"], temp_df["Subtype"])
        print(subtype_table)
        
        # Print stats on the raw probabilities
        if prob_subtype is not None:
            print("\n--- Average Subtype Probability Across Cohort ---")
            for s in range(num_subtypes):
                mean_prob = np.mean(prob_subtype[:, s])
                print(f"  Mean probability of Subtype {s+1}: {mean_prob*100:.1f}%")
        
    # 4. Decoded Sequence of Events for each subtype
    print("\n--- Decoded Atrophy Sequence for Each Subtype ---")
    for s in range(num_subtypes):
        print(f"\nSubtype {s+1} Event Sequence (Healthy -> Severe Atrophy):")
        sequence = ml_sequence_EM[s, :]
        for step, event_idx in enumerate(sequence):
            int_idx = int(event_idx)
            bio_idx = stage_biomarker_index[int_idx]
            z_thresh = stage_zscore[int_idx]
            region_name = regions[bio_idx]
            print(f"  Step {step+1: <2} | {region_name} reaches Z = {z_thresh}")

# Analyze the 2-subtype and 3-subtype models
analyze_pickle("ADNI_asym_subtype1.pickle", num_subtypes=2)
analyze_pickle("ADNI_asym_subtype2.pickle", num_subtypes=3)