import pandas as pd
import numpy as np
from pySuStaIn.ZscoreSustain import ZscoreSustain
import os

# 1. Load your newly created dataset
data_path = "/Users/khoale/Downloads/Alzheimer_Code/csvs/adni_mri_sustain_prepared.csv"
df = pd.read_csv(data_path)

# 2. Define the 12 biomarkers
regions = ["Frontal", "Temporal", "Parietal", "Occipital", "Cingulate", "Insula", 
           "Hippocampus", "Amygdala", "Caudate", "Pallidum", "Putamen", "Accumbens"]
X = df[regions].values

# 3. Define the Z-score thresholds to model as events (Z = 1, 2, and 3 for each region)
Z_vals = np.array([[1, 2, 3]] * len(regions))
Z_max = np.array([5.0] * len(regions))           # Cap maximum Z-score at 5

# 4. Set up fast test parameters (designed to finish in ~10 seconds)
N_startpoints = 5              # 5 starting optimization points (instead of 25)
N_S_max = 1                    # Fit only 1 subtype for testing
N_iterations_MCMC = 1000       # 1,000 MCMC iterations (instead of 1e5)
output_folder = "/Users/khoale/Downloads/Alzheimer_Code/sustain_test_output"
dataset_name = "ADNI_test"

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
print("Starting SuStaIn test run...")
sustain.run_sustain_algorithm()
print("Success! Test run completed. Check output files in 'sustain_test_output'.")