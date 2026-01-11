# 🏥 Diabetes Readmission Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) ![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![Healthcare](https://img.shields.io/badge/Healthcare-FF6B6B?style=for-the-badge)

*Machine learning predictor for 30-day hospital readmissions in diabetic patients. Built with Python, scikit-learn, and UCI hospital data. Features EDA (correlations/heatmaps), logistic regression with interpretable coefficients, and classification reports.*

---

## ✨ Project Overview

This healthcare analytics project, developed as part of a **personal portfolio initiative** in **September 2025**, demonstrates a full ML pipeline for predicting hospital readmissions. The project showcases expertise in:

- 🏊 **Healthcare Analytics** - Medical data analysis
- 🤖 **Machine Learning** - Classification modeling
- 📊 **Feature Engineering** - Clinical feature selection
- 📈 **Model Evaluation** - ROC-AUC, confusion matrix
- 👨‍⚕️ **Clinical Insights** - Actionable recommendations

### 🎯 Key Objectives

- 🔮 **Predict 30-day readmissions** for diabetic patients
- 📉 **Identify high-risk profiles** using clinical features
- 💡 **Provide interpretable models** for healthcare professionals
- 📊 **Analyze correlation patterns** in patient data
- 🎯 **Reduce readmission rates** through early intervention

---

## 📊 What Does It Do?

This Jupyter notebook walks through a full ML pipeline:

### 💾 Data Loading & Cleaning
- Handles **101k+ encounters** from UCI hospital data
- Binarizes target (<30 days readmission = high risk)
- Processes 50+ columns, slimmed to 9 key features

### 🔍 Exploratory Data Analysis (EDA)
- **Correlation heatmaps** and bar plots
- Prior inpatients correlate **−0.15** with risk
- Age, medications, and history analysis

### 🤖 Modeling
- **Logistic regression** with interpretable coefficients
- Key features: age, medications, medical history
- Evaluation: Confusion matrix, classification report, **ROC-AUC**

### 💡 Insights
- Flags **high-risk profiles** (e.g., 27% risk for repeat patients)
- Could **cut readmits 20%** per studies
- Top driver: `number_inpatient` (coef −0.5) – triples odds!

---

## 📊 Key Results

### 📈 Dataset Statistics

- **Dataset:** 101,766 rows, 50 cols → Slimmed to 9 features
- **Class Imbalance:** 11.2% high-risk cases
- **Model Performance:** AUC 0.76, Recall (risky) ≈0.45

### 🎯 Top Findings

- **Top Driver:** `number_inpatient` (coef −0.5) – triples odds!
- **Age Correlation:** Older patients show higher readmission rates
- **Medication Impact:** Number of medications correlates with risk
- **Prior Visits:** Previous inpatient stays are strong predictors

### 📊 Model Metrics

- **AUC Score:** 0.76
- **Recall (High-Risk):** ~0.45
- **Precision:** Balanced for clinical use
- **F1-Score:** Optimized for healthcare applications

---

## 🛠️ Technical Stack

### 💻 Core Technologies

- **Language:** Python 3.8+
- **Environment:** Jupyter Notebook
- **ML Framework:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

### 📚 Key Libraries

| Library | Purpose |
|---------|----------|
| **Pandas** | Data manipulation and cleaning |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Logistic regression and metrics |
| **Matplotlib** | Data visualization |
| **Seaborn** | Correlation heatmaps |

### 🧪 ML Techniques Used

- ✅ **Logistic Regression** - Interpretable coefficients
- ✅ **Feature Selection** - Clinical domain knowledge
- ✅ **Class Imbalance Handling** - Stratified sampling
- ✅ **Cross-Validation** - Robust performance estimation
- ✅ **ROC-AUC Analysis** - Model evaluation

---

## 📦 Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Rishisingh1999/-Diabetes-Readmission-Predictor.git
cd -Diabetes-Readmission-Predictor

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch Jupyter Notebook
jupyter notebook
```

### Google Colab Setup

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook file
3. Upload the dataset (or download from UCI)
4. Run all cells

---

## 🎮 Usage

### Running the Analysis

```python
# Open the notebook
# Execute cells sequentially
# Review visualizations and model results
```

The notebook will:

1. Load and preprocess hospital data
2. Perform exploratory data analysis
3. Engineer clinical features
4. Train logistic regression model
5. Evaluate with confusion matrix and ROC-AUC
6. Generate insights and recommendations

---

## 📊 Visualizations

The project generates:

- 🔥 **Correlation Heatmap** - Feature relationships
- 📉 **Feature Importance** - Bar plots of coefficients
- 📊 **ROC Curve** - Model performance
- 📄 **Confusion Matrix** - Classification results
- 📊 **Distribution Plots** - Patient demographics

![Correlation Heatmap Example](screenshots/corr_heatmap.png)  
*(Screenshot from notebook – add yours!)*

---

## 💼 Business Applications

### 🎯 Use Cases

- **Hospital Administrators:** Resource planning for high-risk patients
- **Clinical Staff:** Early intervention prioritization
- **Care Coordinators:** Discharge planning optimization
- **Health Insurance:** Risk-adjusted premium calculations
- **Policy Makers:** Healthcare quality improvement

### 📈 Healthcare Impact

This solution helps healthcare organizations:

- ✅ **Reduce readmission rates** by 20% through early intervention
- ✅ **Improve patient outcomes** with targeted care
- ✅ **Cut healthcare costs** (readmissions cost $41B annually)
- ✅ **Enhance care quality** with data-driven decisions
- ✅ **Comply with regulations** (CMS readmission penalties)

### 💰 Cost Savings

Hospital readmissions cost the US healthcare system **$41 billion annually**. Reducing readmissions by just 20% could save:

- **$8.2 billion** system-wide savings
- **$10,000+** per prevented readmission
- **Improved patient satisfaction** scores

---

## 📁 Project Structure

```
-Diabetes-Readmission-Predictor/
├── diabetes_readmission.ipynb    # Main analysis notebook
├── data/                          # UCI hospital dataset
│   └── diabetic_data.csv
├── screenshots/                   # Visualization outputs
│   └── corr_heatmap.png
└── README.md                      # Project documentation
```

---

## 🔑 Key Features

### 📊 Clinical Features Used

1. **number_inpatient** - Prior inpatient visits (strongest predictor)
2. **age** - Patient age group
3. **number_medications** - Medication count
4. **num_lab_procedures** - Laboratory tests
5. **num_procedures** - Medical procedures
6. **time_in_hospital** - Length of stay
7. **number_diagnoses** - Diagnosis count
8. **num_medications** - Total medications
9. **medical_specialty** - Admitting specialty

### 🤖 Model Interpretability

- **Coefficient Analysis:** Each feature's impact is quantified
- **Clinical Validation:** Results align with medical literature
- **Actionable Insights:** Clear recommendations for interventions

---

## 🎓 Skills Highlighted

This project demonstrates proficiency in:

- **Healthcare Analytics:** Medical data analysis and interpretation
- **Machine Learning:** Classification modeling and evaluation
- **Python Programming:** Advanced pandas and scikit-learn
- **Statistical Analysis:** Correlation analysis and hypothesis testing
- **Data Visualization:** Heatmaps, ROC curves, confusion matrices
- **Domain Knowledge:** Understanding clinical readmission factors
- **Communication:** Clear documentation for healthcare stakeholders

---

## 🔮 Future Enhancements

- 🤖 **Advanced Models:** XGBoost, Random Forest for better accuracy
- 📊 **Real-time Scoring:** API for live predictions
- 📱 **Web Dashboard:** Interactive Streamlit interface
- 👨‍⚕️ **Clinical Integration:** EMR system integration
- 🔔 **Alert System:** Automated high-risk notifications
- 📊 **A/B Testing:** Intervention effectiveness tracking

---

## 📧 Contact

**Hrushikesh Singh**

- 📧 Email: hrushisingh697@gmail.com
- 💼 LinkedIn: [linkedin.com/in/hrushikesh-singh](https://www.linkedin.com/in/hrushikesh-singh-564b4035a)
- 🐙 GitHub: [@Rishisingh1999](https://github.com/Rishisingh1999)
- 🌐 Portfolio: [rishisingh1999.github.io/my-portfolio-website](https://rishisingh1999.github.io/my-portfolio-website/)

---

## 📄 License

This project is open source and available for educational purposes.

**Attribution appreciated** 🙏

---

## 🙏 Acknowledgments

- **Dataset:** UCI Machine Learning Repository - Diabetes 130-US hospitals dataset
- **Research:** Built on clinical research about readmission risk factors

---

## ⭐ Show Your Support

If you found this project useful, please give it a ⭐ on GitHub!

**Built with ❤️ for Healthcare Analytics & Predictive Medicine**

---
