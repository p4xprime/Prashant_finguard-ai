# 🏦 FinGuard AI
## Bank Credit & Loan Default Risk Analytics Engine

> **IBM SkillsBuild Data Analytics with AI — Academic Internship Program**
> BharatCares & AICTE | **Student: Prashant**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-0f62fe)](https://dataplatform.cloud.ibm.com)
[![License: Academic](https://img.shields.io/badge/License-Academic-green)](https://skillsbuild.org)

---

## 📌 Project Description

FinGuard AI is a production-ready, end-to-end credit risk analytics engine that predicts whether a loan applicant will default on their bank credit obligation. It combines classical machine learning, interactive Plotly visualisations, and IBM watsonx.ai generative AI (Granite) to deliver a comprehensive decision-support platform for credit risk analysts.

**Key Design Principle:** The application **auto-detects all column names, data types, and the target variable** from the uploaded dataset — no column names are hardcoded anywhere in the codebase. The app adapts to the actual Kaggle dataset automatically.

**Domain:** Finance & Banking — Credit Risk & Loan Default Prediction

---

## 📂 Project Structure

```
FinGuard-AI/
├── app.py                         Main Streamlit application (7 tabs)
├── Prashant_FinGuardAI.ipynb      Complete Jupyter Notebook
├── Prashant_ProjectReport.docx    Full project documentation
├── requirements.txt               Python dependencies
├── README.md                      This file
│
└── utils/
    ├── __init__.py
    ├── data_loader.py             Auto-detection engine & preprocessing
    ├── eda.py                     EDA charts and statistics (8 panels)
    ├── model_engine.py            ML training, evaluation, prediction
    ├── watsonx_client.py          IBM watsonx.ai integration
    └── report_generator.py        Excel report generation (7 sheets)
```

---

## 🗄️ Dataset

| Field | Details |
|---|---|
| **Name** | Bank Credit Default — Loan Default Prediction |
| **Source** | Kaggle — kornilovag94 |
| **URL** | https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default |
| **Format** | CSV |
| **Domain** | Banking & Finance — Credit Risk |

> ⚠️ **Dataset Integrity Rule:** This project does NOT fabricate, generate, or assume any dataset structure. Upload the actual Kaggle CSV — the app auto-detects all columns at runtime.

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | ≥ 1.35.0 | Web application framework |
| Pandas | ≥ 2.0.0 | Data manipulation |
| NumPy | ≥ 1.26.0 | Numerical computing |
| Scikit-Learn | ≥ 1.4.0 | ML algorithms & evaluation |
| XGBoost | ≥ 2.0.0 | Gradient boosted trees |
| Plotly | ≥ 5.20.0 | Interactive visualisations |
| OpenPyXL | ≥ 3.1.0 | Excel report export |
| IBM watsonx.ai SDK | ≥ 1.1.0 | Granite LLM integration |
| Jupyter | ≥ 1.0.0 | Notebook environment |

---

## ⚙️ Setup & Run Instructions

### 1. Download the dataset
Download the CSV from Kaggle: https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default

### 2. Clone the repository
```bash
git clone https://github.com/p4xprime/finguard-ai.git
cd finguard-ai
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application
```bash
python -m streamlit run app.py
```
Open your browser at: **http://localhost:8501**

### 6. Run the Jupyter Notebook (alternative)
```bash
jupyter notebook Prashant_FinGuardAI.ipynb
```

---

## 🚀 Features

### 📊 Tab 1 — Data Overview
- Auto-detected column classification (Numeric / Categorical / Binary / Target / ID)
- Target variable analysis and class imbalance severity
- Summary statistics with skewness and kurtosis
- Raw data preview

### 🔍 Tab 2 — Exploratory Data Analysis (8 Panels)
- Target distribution (bar + pie)
- Numeric feature distributions (histogram grid)
- Categorical distributions (top-10 bar charts)
- Correlation heatmap (lower triangle)
- Box plots by target class
- Scatter matrix (top 5 features)
- Missing values analysis
- Outlier detection (IQR method)

### 🤖 Tab 3 — Model Training
- 7 classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, KNN, Naive Bayes
- Stratified train/test split (configurable)
- 5-fold cross-validation
- Grouped metrics comparison chart

### 🏆 Tab 4 — Model Evaluation
- ROC curves (all models overlaid)
- Precision-Recall curves
- Confusion matrix with TP/TN/FP/FN breakdown
- Feature importance (top 25)
- Full classification report

### 🔮 Tab 5 — Risk Predictor
- Dynamic input form auto-generated from actual training features
- Real-time prediction with 5-tier risk classification
- Probability gauge bar
- IBM Granite AI narrative (optional)

### 🤖 Tab 6 — AI Insights (IBM watsonx.ai)
- EDA insights via IBM Granite-13B-Instruct
- Model performance insights via IBM Granite-13B-Instruct

### 📥 Tab 7 — Export Report
- 7-sheet styled Excel report download

---

## 🚦 Risk Tiers

| Probability | Risk Tier |
|---|---|
| ≥ 75% | 🔴 Very High Risk |
| 55–75% | 🟠 High Risk |
| 35–55% | 🟡 Moderate Risk |
| 15–35% | 🟢 Low Risk |
| < 15% | ✅ Very Low Risk |

---

## 🤖 IBM watsonx.ai Setup (Optional)

To enable AI-powered insights:

1. Create an IBM Cloud account: https://cloud.ibm.com
2. Provision IBM watsonx.ai
3. Create an API key: IAM → API Keys → Create
4. Copy your watsonx.ai Project ID
5. Enter credentials in the FinGuard AI sidebar under **IBM watsonx.ai**

**Model used:** `ibm/granite-13b-instruct-v2`

---

## 📁 Submitted Files

| File | Format | Description |
|---|---|---|
| `Prashant_FinGuardAI.ipynb` | `.ipynb` | Complete project code (Jupyter Notebook) |
| `requirements.txt` | `.txt` | Python dependencies list |
| `Prashant_ProjectReport.docx` | `.docx` | Full project documentation (15 sections) |
| `README.md` | `.md` | This file — project overview |

---

## 🎓 Academic Information

| Field | Details |
|---|---|
| Program | IBM SkillsBuild Data Analytics with AI |
| Internship Provider | BharatCares |
| Approved By | AICTE (All India Council for Technical Education) |
| Student | Prashant |
| Project Title | FinGuard AI — Bank Credit and Loan Default Risk Analytics Engine |
| GitHub | https://github.com/p4xprime/finguard-ai |

---

## 📝 License

This project is developed for academic purposes as part of the IBM SkillsBuild internship program.  
Dataset is sourced from Kaggle under its respective terms of service.
