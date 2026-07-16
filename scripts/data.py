import pandas as pd

# 1. Load your original ADNI CSV
csv_path = '/Users/khoale/Downloads/Alzheimer_Code/T1-MRI_and_or_FDG-PET_4_07_2026.csv'  # Using your actual CSV name
df = pd.read_csv(csv_path)

# Convert dates to datetime objects for math
df['Acq Date'] = pd.to_datetime(df['Acq Date'])

# 2. Separate MRI and PET modalities
mri_df = df[df['Modality'] == 'MRI']
pet_df = df[df['Modality'] == 'PET'].copy()

# Filter MRIs to keep only T1-weighted images based on description
mri_t1 = mri_df[mri_df['Description'].str.contains('MPRAGE|SPGR|T1', case=False, na=False)].copy()

# Remove duplicate scans taken on the exact same day for the same subject
mri_t1 = mri_t1.sort_values(['Subject', 'Acq Date']).drop_duplicates(subset=['Subject', 'Acq Date'])
pet_df = pet_df.sort_values(['Subject', 'Acq Date']).drop_duplicates(subset=['Subject', 'Acq Date'])

# 3. Find valid pairs within 1 year (365 days) WITH STRICT DIAGNOSIS MATCHING
pairs = []

# Get a list of unique subjects to iterate through
unique_subjects = mri_t1['Subject'].unique()

for subj in unique_subjects:
    # Get all T1 MRIs for this specific subject
    subj_mris = mri_t1[mri_t1['Subject'] == subj]
    
    # Flag to track if we found a valid pair for this subject
    found_pair_for_subject = False
    
    for idx, mri_row in subj_mris.iterrows():
        mri_date = mri_row['Acq Date']
        mri_group = mri_row['Group']  # Diagnosis at the time of MRI
        
        # STRICT RULE 1: Find PET scans for this subject that share the EXACT same diagnosis
        pet_cands = pet_df[(pet_df['Subject'] == subj) & (pet_df['Group'] == mri_group)].copy()
        
        if len(pet_cands) > 0:
            # Calculate absolute time difference in days
            pet_cands['Days_Diff'] = (pet_cands['Acq Date'] - mri_date).dt.days.abs()
            
            # STRICT RULE 2: Must be within a year (365 days)
            valid_pets = pet_cands[pet_cands['Days_Diff'] <= 365]
            
            if len(valid_pets) > 0:
                # We found a valid pair! Pick the closest one in time
                closest_pet = valid_pets.loc[valid_pets['Days_Diff'].idxmin()]
                
                pairs.append({
                    'Subject': subj,
                    'Group': mri_group,
                    'MRI_ID': mri_row['Image Data ID'],
                    'MRI_Date': mri_date.strftime('%Y-%m-%d'),
                    'PET_ID': closest_pet['Image Data ID'],
                    'PET_Date': closest_pet['Acq Date'].strftime('%Y-%m-%d'),
                    'Days_Diff': closest_pet['Days_Diff']
                })
                
                # Mark that we successfully paired this subject and stop looking for more pairs for them
                found_pair_for_subject = True
                break  # Break out of the MRI loop for this subject
                
    # If we want a maximum of 500 pairs
    if len(pairs) == 500: 
        break

# Convert pairs into a DataFrame
pairs_df = pd.DataFrame(pairs)

# 4. Save results
if len(pairs_df) > 0:
    pairs_df.to_csv('strict_mri_pet_pairs.csv', index=False)
    
    # Extract unique subjects that survived the strict filtering
    final_subjects = pairs_df['Subject'].unique()
    
    with open('strict_valid_subject_ids.txt', 'w') as f:
        for subj in final_subjects:
            f.write(f"{subj}, ")
            
    print(f"Success! Found {len(pairs_df)} strictly paired subjects.")
    print("These subjects have an MRI and a PET within 365 days while holding the EXACT same diagnosis.")
else:
    print("No subjects found matching the strict criteria.")


    