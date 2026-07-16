# Preprocessing Pipeline for pySuStaIn on ADNI MRI Dataset

This document details the exact steps performed to preprocess the raw FreeSurfer T1-MRI volumetric dataset ([adni_mri_ucsf_merged.csv](file:///Users/khoale/Downloads/Alzheimer_Code/adni_mri_ucsf_merged.csv)) into a cleaned, 12-biomarker, positive z-score dataset suitable for running `ZscoreSustain` ([adni_mri_sustain_prepared.csv](file:///Users/khoale/Downloads/Alzheimer_Code/adni_mri_sustain_prepared.csv)).

---

## Preprocessing Steps

### Step 1: Feature Identification and Lobar Aggregation
The raw FreeSurfer dataset contains 326 structural measurement (`ST`) columns representing fine-grained sub-regions of both left and right hemispheres (e.g., surface areas, cortical thicknesses, and regional volumes).

To make the SuStaIn model computationally feasible and biologically interpretable (following the original SuStaIn publication), we aggregated these columns into **12 bilateral brain regions** (6 cortical lobes + 6 deep subcortical structures) by summing corresponding Left and Right hemisphere volumes:

* **Cortical Lobes (Gray Matter Volumes):**
  * **Frontal Lobe:** Sum of 22 bilateral FreeSurfer cortical volume (`CV`) columns (including precentral, superior frontal, middle frontal, orbital, and pars regions).
  * **Temporal Lobe:** Sum of 18 bilateral `CV` columns (including middle temporal, superior temporal, fusiform, and entorhinal regions).
  * **Parietal Lobe:** Sum of 10 bilateral `CV` columns (including postcentral, superior parietal, inferior parietal, supramarginal, and precuneus).
  * **Occipital Lobe:** Sum of 8 bilateral `CV` columns (including lateral occipital, lingual, cuneus, and pericalcarine).
  * **Cingulate Cortex:** Sum of 8 bilateral `CV` columns (anterior, posterior, and isthmus cingulate).
  * **Insula Cortex:** Sum of bilateral insula gray matter volumes.
* **Subcortical Structures (Bilateral Volumes):**
  * **Hippocampus** (`ST29SV` + `ST88SV`)
  * **Amygdala** (`ST12SV` + `ST71SV`)
  * **Caudate** (`ST16SV` + `ST75SV`)
  * **Pallidum** (`ST42SV` + `ST101SV`)
  * **Putamen** (`ST53SV` + `ST112SV`)
  * **Nucleus Accumbens** (`ST11SV` + `ST70SV`)

---

### Step 2: Quality Control and Head Size Correction (ICV Normalization)
Because absolute brain volumes are heavily influenced by natural variations in head size, we normalized all volumes against the **Total Intracranial Volume (ICV)** (represented by column `ST10CV`) using residuals:
1. Checked and converted all columns to numeric, replacing quality control failure strings (such as "Failed") with `NaN`.
2. Excluded subjects missing `ST10CV` (ICV).
3. Filtered out the Cognitively Normal (CN) cohort to fit a baseline regression line:
   $$\text{Region Volume} = \beta \times \text{ICV} + \alpha$$
4. For all subjects in the dataset, we calculated their **residual volume** (actual volume minus predicted volume based on head size):
   $$\text{Residual} = \text{Actual Volume} - (\beta \times \text{ICV} + \alpha)$$
5. Added back the mean volume of the healthy controls to keep the corrected volumes on a realistic biological scale:
   $$\text{Corrected Volume} = \text{Residual} + \mu_{\text{CN}}$$

---

### Step 3: Z-Scoring and Positive Atrophy Inversion
SuStaIn models disease progression as sequential positive deviations from normality. Because structural gray matter volumes **shrink** (atrophy) with disease progression, a standard z-score would become negative:
$$Z_{\text{standard}} = \frac{X - \mu_{\text{CN}}}{\sigma_{\text{CN}}} \quad (\text{negative for atrophy})$$

To align with SuStaIn's expectation of positive values for disease abnormalities, we inverted the z-score calculation:
$$Z_{\text{SuStaIn}} = \frac{\mu_{\text{CN}} - X}{\sigma_{\text{CN}}}$$
This maps a volume reduction (atrophy) to an increasing **positive** standard deviation value.

---

### Step 4: Missing Value Imputation and Zero-Clipping
1. MCMC algorithms like SuStaIn cannot process missing values (`NaN`). Any missing regional volume measurements (from FreeSurfer segmentation failures) were imputed with `0.0` (indicating a normal, non-atrophied region).
2. Negative z-scores (where a subject's regional volume is larger/healthier than the control baseline mean) were clipped (clamped) to `0.0`. This ensures that healthy variation does not interfere with the progression model.
   $$\text{Final } Z = \max(0.0, Z_{\text{SuStaIn}})$$

---

## Final Output File Format

The resulting file **[adni_mri_sustain_prepared.csv](file:///Users/khoale/Downloads/Alzheimer_Code/adni_mri_sustain_prepared.csv)** has the following properties:
* **Dimensions:** 577 rows, 14 columns.
* **Identifiers:** `PTID` (subject ID) and `Label` (disease classification).
* **Biomarkers:** 12 numerical columns (`Frontal`, `Temporal`, `Parietal`, `Occipital`, `Cingulate`, `Insula`, `Hippocampus`, `Amygdala`, `Caudate`, `Pallidum`, `Putamen`, `Accumbens`) representing corrected, positive z-scores.
* **Range:** Minimum value is strictly `0.0` across all features, with maximums up to `14.3` standard deviations of atrophy in diseased patients.
