# Diabetes 30-Day Readmission Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)

Machine-learning prototype that predicts **30-day hospital readmission** for diabetic patients using the publicly available UCI Diabetes 130-US Hospitals dataset (1999–2008).

> **Important**: This is an educational portfolio project. It is **not** a clinical decision-support tool and has not undergone prospective validation.

---

## 1. Project Purpose

Hospital readmissions within 30 days are costly and often preventable. This project builds a transparent, reproducible baseline model that:

- Predicts whether a diabetic patient will be readmitted within 30 days of discharge.
- Uses only features that are typically available at discharge time.
- Emphasizes patient-grouped validation to avoid leakage across repeated encounters of the same patient.

---

## 2. Dataset Source & Attribution

| Item | Value |
|------|-------|
| **Primary source** | [UCI ML Repository – Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| **Kaggle mirror** | [brandao/diabetes](https://www.kaggle.com/datasets/brandao/diabetes) |
| **File used** | `diabetic_data.csv` |
| **Rows / Columns** | 101,766 encounters × 50 columns |
| **Unique patients** | 71,518 |
| **SHA-256** | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| **Access date** | 2026-10-27 |

**Target definition (explicit)**  
`target = 1` if `readmitted == '<30'`, else `0` (i.e., both `NO` and `>30` are treated as negative).  
Class prevalence of the positive class: **11.16 %**.

---

## 3. Data Cleaning & Feature Engineering

- Age is mapped from the original ordinal bins (`[0-10)` … `[90-100)`) to integers 0–9.
- Nine numeric clinical features available at discharge are retained:
  - `age`, `time_in_hospital`, `num_lab_procedures`, `num_procedures`,
  - `num_medications`, `number_outpatient`, `number_emergency`,
  - `number_inpatient`, `number_diagnoses`
- No medication-change or laboratory-result features that could introduce leakage are used.
- Missing values in the selected numeric columns are negligible; no imputation is required for this feature set.

---

## 4. Model & Validation Methodology

- **Split**: `GroupShuffleSplit` by `patient_nbr` (20 % held-out test, then 20 % of remaining as validation). This guarantees that all encounters of the same patient stay inside one fold.
- **Model**: `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)` with `StandardScaler`.
- **Threshold selection**: F1-score maximised on the **validation** set only; the chosen threshold is then applied to the untouched test set.
- **Metrics reported on the held-out test set**: ROC-AUC, Average Precision, Precision, Recall, F1, Confusion Matrix.

---

## 5. Actual Test-Set Results (run 2026-10-27)

| Metric | Value @ thr=0.5 | Value @ tuned thr (≈0.49) |
|--------|-----------------|---------------------------|
| ROC-AUC | 0.629 | 0.629 |
| Average Precision | 0.180 | 0.180 |
| Precision | 0.161 | 0.155 |
| Recall | 0.460 | 0.501 |
| F1 | 0.239 | 0.237 |

**Confusion matrix (thr = 0.5)**

```
[[12864  5138]
 [ 1162   989]]
```

**Top coefficients (standardized features)**  
`number_inpatient` (+0.38) is the strongest positive driver, followed by age and number of medications. This aligns with clinical literature that prior inpatient stays are a major risk factor.

All numbers above are taken from a fresh run of `run_experiment.py` on the exact dataset file whose hash is recorded above. Full machine-readable metrics live in `outputs/metrics.json`.

---

## 6. Generated Artifacts

```
outputs/
├── metrics.json              # full run configuration + metrics
├── test_predictions.csv      # y_true, y_proba, predictions
├── correlation_heatmap.png
├── feature_coefficients.png
├── roc_curve.png
├── pr_curve.png
└── confusion_matrix.png
```

---

## 7. Setup & Reproduction

```bash
# 1. Clone
git clone https://github.com/Rishisingh1999/-Diabetes-Readmission-Predictor.git
cd -Diabetes-Readmission-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Obtain the dataset
#    Option A – Kaggle CLI (recommended)
kaggle datasets download -d brandao/diabetes -p data --unzip
#    Option B – Download diabetic_data.csv from the UCI page and place it
#               in the project root or in data/

# 4. Run the experiment
python run_experiment.py
```

The script writes all plots and the metrics file into `outputs/`.

---

## 8. Limitations & Known Risks

- Historical data (1999–2008); care patterns and coding practices have changed.
- Strong class imbalance; high recall comes at the cost of many false positives.
- Only a modest feature set was used; richer models (or embedding of ICD codes) can improve performance.
- No prospective clinical validation has been performed.
- The model must **never** be used for real clinical decision-making.

---

## 9. Project Structure

```
-Diabetes-Readmission-Predictor/
├── run_experiment.py          # fully reproducible training + evaluation
├── requirements.txt
├── README.md
├── .gitignore
├── data/                      # place diabetic_data.csv here (git-ignored)
└── outputs/                   # generated by the experiment (committed for transparency)
```

---

## License & Attribution

Educational use. Please attribute the original UCI dataset authors (Strack et al).
