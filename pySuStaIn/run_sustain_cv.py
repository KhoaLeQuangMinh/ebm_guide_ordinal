import pandas as pd
import numpy as np
import sklearn.model_selection
from pySuStaIn.ZscoreSustain import ZscoreSustain
import os

# 1. Load the prepared dataset
data_path = "/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_sustain_prepared.csv"
df = pd.read_csv(data_path)

# 2. Select the 12 biomarkers
regions = ["Frontal", "Temporal", "Parietal", "Occipital", "Cingulate", "Insula", 
           "Hippocampus", "Amygdala", "Caudate", "Pallidum", "Putamen", "Accumbens"]
X = df[regions].values

# 3. Setup SuStaIn thresholds (Z-scores of 1, 2, 3 for each region)
Z_vals = np.array([[1, 2, 3]] * len(regions))
Z_max = np.array([5.0] * len(regions))

# 4. Configure Cross-Validation Parameters
N_folds = 5                    # 5-fold cross-validation
N_startpoints = 25             # Ensure robust local-minimum search
N_S_max = 3                    # Compare 1, 2, and 3 subtypes
N_iterations_MCMC = 10000      # 10,000 MCMC iterations per fold (fast but robust for CV)
output_folder = "/Users/khoale/Downloads/Alzheimer_Code/sustain_cv_output"
dataset_name = "ADNI_MRI"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 5. Initialize ZscoreSustain
sustain = ZscoreSustain(
    data=X,
    Z_vals=Z_vals,
    Z_max=Z_max,
    biomarker_labels=regions,
    N_startpoints=N_startpoints,
    N_S_max=N_S_max,
    N_iterations_MCMC=N_iterations_MCMC,
    output_folder=output_folder,
    dataset_name=dataset_name,
    use_parallel_startpoints=True
)

# 6. Generate cross-validation indices
test_idxs = []
cv = sklearn.model_selection.KFold(n_splits=N_folds, shuffle=True, random_state=42)
for train, test in cv.split(X):
    test_idxs.append(test)

print(f"Starting {N_folds}-fold Cross-Validation...")
print("This will run SuStaIn 5 times. It may take 15 to 30 minutes to complete.")

# 7. Run the CV model
CVIC, loglike_matrix = sustain.cross_validate_sustain_model(test_idxs, plot=True)

# 8. Print CVIC results (Lowest CVIC represents the best model)
print("\n" + "="*50)
print(" CROSS-VALIDATION SUMMARY ".center(50, "="))
print("="*50)
for s in range(N_S_max):
    print(f"  {s+1} Subtype Model CVIC: {CVIC[s]:.1f}")
print("="*50)
print("Rule of thumb: The model with the LOWEST CVIC is the optimal model.")