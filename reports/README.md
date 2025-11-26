# Heart Disease Dataset – Exploratory Data Analysis & Modeling Report

## 1. Dataset overview

- **File**: `data/heart.csv`
- **Rows**: 918
- **Columns**: 12
- **Target variable**: `HeartDisease`
  - 1 = presence of heart disease
  - 0 = no heart disease

### 1.1 Features

- **Numeric**
  - `Age`: 28–77 years  
    - Mean ≈ 53.5  
    - Std ≈ 9.4
  - `RestingBP`: 0–200 mmHg  
    - Mean ≈ 132.4
  - `Cholesterol`: 0–603 mg/dl  
    - Mean ≈ 198.8
  - `FastingBS`: 0 or 1 (treated as numeric)
  - `MaxHR`: 60–202 bpm  
    - Mean ≈ 136.8
  - `Oldpeak`: -2.6–6.2  
    - Mean ≈ 0.89
- **Categorical**
  - `Sex`: {M, F}
  - `ChestPainType`: {ASY, NAP, ATA, TA}
  - `RestingECG`: {Normal, ST, LVH}
  - `ExerciseAngina`: {Y, N}
  - `ST_Slope`: {Up, Flat, Down}

### 1.2 Data quality

- **Missing values**:  
  - No missing values in any column (all columns have 0 NaNs).
- **Potential anomalies**:
  - Some values of `RestingBP = 0` and `Cholesterol = 0` are physiologically implausible and likely represent missing/erroneous records encoded as 0. This should be considered in more advanced cleaning.

---

## 2. Target variable analysis

### 2.1 Class distribution

`HeartDisease`:

- 1 (disease): **508** cases (~55.3%)
- 0 (no disease): **410** cases (~44.7%)

The dataset is slightly imbalanced towards presence of heart disease but still reasonably balanced; standard metrics like accuracy are still meaningful.

---

## 3. Univariate analysis

### 3.1 Numerical features

**Age**

- Mean ≈ 53.5 years; most patients lie roughly between late 40s and early 60s.
- Heart disease is more frequent in older age brackets.

**RestingBP**

- Mean ≈ 132 mmHg, with many values in the pre-hypertensive and hypertensive range.
- Some zeros exist (outliers/invalid).

**Cholesterol**

- Mean ≈ 199 mg/dl, with a wide spread up to 603 mg/dl.
- Some zeros (likely missing/erroneous).

**MaxHR**

- Mean ≈ 137 bpm, range 60–202.
- Higher MaxHR tends to be more common in non-disease cases.

**Oldpeak**

- Mean ≈ 0.89, range -2.6 to 6.2.
- Higher values of Oldpeak correspond to more ST depression and are associated with heart disease.

### 3.2 Categorical features

Distributions (counts):

- **Sex**
  - M: 725
  - F: 193

- **ChestPainType**
  - ASY: 496
  - NAP: 203
  - ATA: 173
  - TA: 46

- **RestingECG**
  - Normal: 552
  - LVH: 188
  - ST: 178

- **ExerciseAngina**
  - N: 547
  - Y: 371

- **ST_Slope**
  - Flat: 460
  - Up: 395
  - Down: 63

---

## 4. Bivariate analysis – Relationship with HeartDisease

### 4.1 Numeric vs HeartDisease (correlations)

Correlation of numeric features with `HeartDisease`:

- `Oldpeak`: **+0.40** (strong positive)
- `Age`: **+0.28**
- `FastingBS`: **+0.27**
- `RestingBP`: **+0.11**
- `Cholesterol`: **-0.23**
- `MaxHR`: **-0.40**

Key points:

- Higher **Oldpeak** (more ST depression) strongly increases the likelihood of heart disease.
- Older **Age** and higher **FastingBS** are associated with higher heart disease risk.
- Higher **MaxHR** is **negatively** correlated with heart disease: patients who can reach higher max heart rates are less likely to have heart disease.
- Interestingly, **Cholesterol** shows a **negative** correlation with heart disease in this dataset. This might reflect treatment effects or dataset quirks (e.g., healthier patients not on statins having higher recorded cholesterol, or measurement/selection bias).

### 4.2 Categorical vs HeartDisease

Using row-normalized crosstabs (proportion of heart disease within each category):

#### Sex

- **Female (F)**:  
  - No disease: ~74.1%  
  - Disease: ~25.9%
- **Male (M)**:  
  - No disease: ~36.8%  
  - Disease: ~63.2%

> 🔍 **Insight**: Males in this dataset have a **much higher** prevalence of heart disease than females.

#### ChestPainType

- **ASY (asymptomatic)**:  
  - No disease: ~21.0%  
  - Disease: ~79.0%
- **ATA (typical angina)**:  
  - No disease: ~86.1%  
  - Disease: ~13.9%
- **NAP (non-anginal pain)**:  
  - No disease: ~64.5%  
  - Disease: ~35.5%
- **TA**:  
  - No disease: ~56.5%  
  - Disease: ~43.5%

> 🔍 **Insight**:
> - **Asymptomatic** chest pain is very strongly associated with heart disease.
> - **Typical angina (ATA)** appears mostly in patients without diagnosed heart disease in this dataset.

#### ExerciseAngina

- **N**:
  - No disease: ~64.9%
  - Disease: ~35.1%
- **Y**:
  - No disease: ~14.8%
  - Disease: ~85.2%

> 🔍 **Insight**: Exercise-induced angina (`Y`) is a very strong indicator of heart disease.

#### ST_Slope

- **Up**:
  - No disease: ~80.3%
  - Disease: ~19.7%
- **Flat**:
  - No disease: ~17.2%
  - Disease: ~82.8%
- **Down**:
  - No disease: ~22.2%
  - Disease: ~77.8%

> 🔍 **Insight**:
> - **Flat** and **Down** ST slopes are highly associated with heart disease.
> - **Up** slope is strongly associated with absence of heart disease (protective pattern).

#### RestingECG

- **Normal**:
  - No disease: ~48.4%
  - Disease: ~51.6%
- **LVH**:
  - No disease: ~43.6%
  - Disease: ~56.4%
- **ST**:
  - No disease: ~34.3%
  - Disease: ~65.7%

> 🔍 **Insight**:
> - ST-type ECG patterns (ST, LVH) are somewhat more skewed towards heart disease than Normal ECGs, especially ST.

---

## 5. Modeling

### 5.1 Model setup

- **Algorithm**: Logistic Regression (baseline linear classifier)
- **Preprocessing**:
  - Numeric features: standardized with `StandardScaler`
  - Categorical features: one-hot encoded with `OneHotEncoder`
- **Train-test split**:
  - 80% train, 20% test
  - Stratified by `HeartDisease`
  - `random_state = 42`

### 5.2 Performance metrics

On the held-out test set (20%):

- **Accuracy**: ~0.886
- **ROC-AUC**: ~0.93

**Class-wise performance (approximate):**

- Class 0 (no heart disease):
  - Precision ≈ 0.91
  - Recall ≈ 0.83
  - F1 ≈ 0.87
- Class 1 (heart disease):
  - Precision ≈ 0.87
  - Recall ≈ 0.93
  - F1 ≈ 0.90

> ✅ **Conclusion**: The baseline Logistic Regression model performs strongly, with high recall for the positive class (heart disease) and good overall discrimination (AUC ≈ 0.93).

### 5.3 Feature importance (Logistic Regression coefficients)

Top **positive** coefficients (risk-increasing):

- `ChestPainType_ASY`
- `ST_Slope_Flat`
- `Sex_M`
- `ExerciseAngina_Y`
- `FastingBS`
- `Oldpeak`
- `RestingECG_LVH`
- `ST_Slope_Down`

Top **negative** coefficients (risk-decreasing or associated with no disease):

- `ST_Slope_Up`
- `ChestPainType_NAP`
- `Sex_F`
- `Cholesterol`
- `ExerciseAngina_N`
- `ChestPainType_ATA`
- `MaxHR`

> 🔍 **Interpretation**:
> - Being **male**, having **asymptomatic chest pain**, **exercise-induced angina**, **flat/down ST slopes**, higher **Oldpeak**, and **elevated fasting blood sugar** are strong risk indicators.
> - **Up-sloping ST**, **female sex**, **typical angina (ATA)**, **no exercise angina**, and **higher MaxHR** are associated with lower risk in this dataset.

---

## 6. Key insights & recommendations

1. **High-risk profiles** in this dataset:
   - Older male patients
   - Asymptomatic chest pain (ASY)
   - Exercise-induced angina (Y)
   - Flat or down-sloping ST segment
   - Higher Oldpeak
   - Elevated fasting blood sugar

2. **Protective patterns**:
   - Up-sloping ST segment
   - Higher MaxHR achieved during exercise
   - Typical angina (ATA) appears less related to diagnosed heart disease in this dataset.

3. **Data quality considerations**:
   - `RestingBP = 0` and `Cholesterol = 0` are likely invalid and could be treated as missing in a more refined analysis.
   - Further work could:
     - Impute or remove such rows
     - Explore non-linear models (Random Forest, XGBoost)
     - Use cross-validation for more robust evaluation

4. **Model usefulness**:
   - Even a simple Logistic Regression with basic preprocessing achieves **strong performance** (AUC ≈ 0.93).
   - This model can be the baseline for comparison against more complex models.

---


