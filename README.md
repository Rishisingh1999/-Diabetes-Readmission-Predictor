📊 Project Overview
This Jupyter notebook walks through a full ML pipeline:

Data Loading & Cleaning: Handles 101k+ encounters, binarizes target (<30 days readmission = high risk).
EDA: Correlation heatmaps, bar plots – e.g., prior inpatients correlate ~0.15 with risk.
Modeling: Logistic regression (interpretable coefs) on key features like age, meds, history.
Evaluation: Confusion matrix, classification report, ROC-AUC.
Insights: Flags high-risk profiles (e.g., 27% risk for repeats) – could cut readmits 20% per studies.

Key Results

Dataset: 101,766 rows, 50 cols → Slimmed to 9 features.
Imbalance: 11.2% high-risk cases.
Model Perf: AUC 0.76, Recall (risky) ~0.45.
Top Driver: number_inpatient (coef ~0.5) – triples odds!

<img src="screenshots/corr_heatmap.png" alt="Corr Heatmap Example">
(Screenshot from notebook – add yours!)
🛠️ Tech Stack

Language: Python 3.8+
ML: Scikit-learn (LogisticRegression, metrics)
Data: Pandas, NumPy
Viz: Matplotlib, Seaborn (heatmaps, bars)
Notebook: Jupyter

📈 How to Extend

Tune Model: Add class_weight='balanced' or try RandomForest.
More Features: Include race/gender (with ethics checks).
Deploy: Wrap in Streamlit for a web app – predict on new patients.

🤝 Contributing
Fork it, PRs welcome! Issues for bugs/features. Let's make healthcare AI better.
📄 License
MIT – Free to use/modify/share.
Acknowledgments

Dataset: UCI Machine Learning Repository
Inspired by: Healthcare analytics papers on readmission reduction.
