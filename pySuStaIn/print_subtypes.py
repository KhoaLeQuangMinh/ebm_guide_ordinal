import pandas as pd
# Load the exported results
df = pd.read_csv("/Users/khoale/Downloads/Alzheimer_Code/sustain_subject_staging_results_as.csv")

def get_stage_phase(stage):
    if stage == 0:
        return "Stage 0 (No Atrophy)"
    elif 1 <= stage <= 16:
        return "Stages 1-16 (Early)"
    elif 17 <= stage <= 32:
        return "Stages 17-32 (Moderate)"
    else:
        return "Stages 33-48 (Severe)"

df["Disease_Phase"] = df["Assigned_Stage"].apply(get_stage_phase)
phase_order = ["Stage 0 (No Atrophy)", "Stages 1-16 (Early)", "Stages 17-32 (Moderate)", "Stages 33-48 (Severe)"]

subtype_names = {
    1: "Subtype 1: Subcortical-First",
    2: "Subtype 2: Typical AD (Limbic-First)",
    3: "Subtype 3: Asymmetric Left-Hemisphere Dominant"
}

for s in [1, 2, 3]:
    sub_df = df[df["Assigned_Subtype"] == s]
    print(f"\n==================== {subtype_names[s]} (n = {len(sub_df)}) ====================")
    
    # Generate crosstab
    ct = pd.crosstab(sub_df["Label"], sub_df["Disease_Phase"])
    
    # Print row by row for readability
    for label in ["CN", "sMCI", "pMCI", "AD"]:
        if label in ct.index:
            print(f"\nDiagnosis: {label}")
            for phase in phase_order:
                val = ct.loc[label, phase] if phase in ct.columns else 0
                print(f"  - {phase: <25}: {val} subjects")