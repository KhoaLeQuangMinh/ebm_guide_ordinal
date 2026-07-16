import pickle
import pandas as pd
import numpy as np

# 1. Path to your dataset and the generated pickle file
data_path = "/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_sustain_prepared.csv"
# Change this path to point to your actual pickle file in the output folder!
pickle_path = "/Users/khoale/Downloads/Alzheimer_Code/sustain_test_output/pickle_files/ADNI_test_subtype0.pickle"

# 2. Load dataset and pickle file
df = pd.read_csv(data_path)

with open(pickle_path, 'rb') as f:
    results = pickle.load(f)

# Extract key variables
ml_subtype = results["ml_subtype"]  # Subtype assignment (0, 1, 2, ...)
ml_stage = results["ml_stage"]      # Stage assignment (0, 1, 2, ...)

# Add results back to the dataframe
df["Inferred_Subtype"] = ml_subtype
df["Inferred_Stage"] = ml_stage

print("--- DIAGNOSTIC CHECK 1: COHORT SUMMARY ---")
print(f"Total subjects loaded: {len(df)}")
unique_subtypes = np.unique(ml_subtype)
print(f"Number of subtypes fitted: {len(unique_subtypes)}")

print("\n--- DIAGNOSTIC CHECK 2: CLINICAL LABEL vs STAGE DISTRIBUTION ---")
# Pivot table showing how many subjects of each diagnosis label are in each stage
stage_crosstab = pd.crosstab(df["Label"], df["Inferred_Stage"])
print(stage_crosstab)

print("\n--- DIAGNOSTIC CHECK 3: SUBTYPE FRACTIONS ---")
# Calculate the percentage of subjects assigned to each subtype
if len(unique_subtypes) > 1:
    subtype_crosstab = pd.crosstab(df["Label"], df["Inferred_Subtype"])
    print(subtype_crosstab)
else:
    print("Only 1 subtype was fitted (test run). Proportion checks are relevant for 2+ subtypes.")

# Check convergence diagnostic
likelihoods = results["samples_likelihood"]
print("\n--- DIAGNOSTIC CHECK 4: CONVERGENCE TRACE STATS ---")
print(f"First 10 MCMC likelihoods: {likelihoods[:10].flatten()}")
print(f"Last 10 MCMC likelihoods: {likelihoods[-10:].flatten()}")
# If the mean of the last 10% is significantly higher than the first 10%, it hasn't converged
first_half_mean = np.mean(likelihoods[:len(likelihoods)//2])
second_half_mean = np.mean(likelihoods[len(likelihoods)//2:])
print(f"First half mean log-likelihood: {first_half_mean:.3f}")
print(f"Second half mean log-likelihood: {second_half_mean:.3f}")