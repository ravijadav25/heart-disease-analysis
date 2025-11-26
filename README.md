# Heart Disease Prediction – EDA & Modeling Project

This repository contains an end-to-end data analysis project on a heart disease dataset (`heart.csv`).
It includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- A baseline Logistic Regression model
- A written analysis report with key insights

## Dataset

- File: `data/heart.csv`
- Rows: 918
- Columns: 12
- Target: `HeartDisease` (1 = heart disease, 0 = no heart disease)

### Features

- **Numerical**
  - `Age` – Age in years
  - `RestingBP` – Resting blood pressure (mm Hg)
  - `Cholesterol` – Serum cholesterol (mg/dl)
  - `FastingBS` – Fasting blood sugar (> 120 mg/dl) (1 = true; 0 = false)
  - `MaxHR` – Maximum heart rate achieved
  - `Oldpeak` – ST depression induced by exercise relative to rest
- **Categorical**
  - `Sex` – M, F
  - `ChestPainType` – ATA, NAP, ASY, TA
  - `RestingECG` – Normal, ST, LVH
  - `ExerciseAngina` – Y, N
  - `ST_Slope` – Up, Flat, Down

## Project Structure

```text
.
├── data/
│   └── heart.csv
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── eda.py
│   └── modeling.py
├── reports/
│   └── heart_disease_analysis_report.md
├── README.md
└── requirements.txt
