import pandas as pd
import numpy as np
from pySuStaIn.ZscoreSustain import ZscoreSustain
import os

# 1. Load the asymmetric dataset
data_path = "/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_sustain_prepared_asymmetric_as.csv"
df = pd.read_csv(data_path)

# 2. Define the 24 asymmetric biomarkers
regions = [
    "L_Frontal", "L_Temporal", "L_Parietal", "L_Occipital", "L_Cingulate", "L_Insula",
    "L_Hippocampus", "L_Amygdala", "L_Caudate", "L_Pallidum", "L_Putamen", "L_Accumbens",
    "R_Frontal", "R_Temporal", "R_Parietal", "R_Occipital", "R_Cingulate", "R_Insula",
    "R_Hippocampus", "R_Amygdala", "R_Caudate", "R_Pallidum", "R_Putamen", "R_Accumbens"
]
X = df[regions].values

# 3. Setup SuStaIn thresholds (Z-scores of 1 and 2 for each region)
Z_vals = np.array([[1, 2]] * len(regions))       # Shape: 24 biomarkers x 2 thresholds
Z_max = np.array([5.0] * len(regions))           # Cap maximum Z-score at 5.0

# 4. Configure production parameters
N_startpoints = 25              # Standard for global search
N_S_max = 3                     # Fits 1, 2, and 3 subtypes
N_iterations_MCMC = 100000      # 100,000 iterations for MCMC
output_folder = "/Users/khoale/Downloads/Alzheimer_Code/sustain_asymmetric_as_output"
dataset_name = "ADNI_asym"

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

# 6. Run the model
print("Starting Asymmetric SuStaIn production run (Z = 1, 2)...")
print("This will fit 1, 2, and 3 subtypes. It may take 15 to 30 minutes to complete.")
sustain.run_sustain_algorithm()
print("Success! Asymmetric run complete. Results saved in 'sustain_production_output_asymmetric'.")