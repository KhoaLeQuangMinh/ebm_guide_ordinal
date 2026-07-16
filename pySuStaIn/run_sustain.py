import pandas as pd
import numpy as np
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

# 4. Configure production-level parameters
N_startpoints = 25              # Standard for finding the global optimum
N_S_max = 3                     # Fits 1, 2, and 3 subtypes sequentially
N_iterations_MCMC = 100000      # 100,000 iterations for stable uncertainty estimation
output_folder = "/Users/khoale/Downloads/Alzheimer_Code/sustain_output"
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

# 6. Run the production model
print("Starting SuStaIn production run...")
print("This will fit 1, 2, and 3 subtypes. It may take 10 to 20 minutes to complete.")
sustain.run_sustain_algorithm()
print("Success! Production run complete. Results saved in 'sustain_production_output'.")