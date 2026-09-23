# 🏦 FinGuard AI
## Bank Credit & Loan Default Risk Analytics Engine

> **IBM SkillsBuild Data Analytics with AI — Academic Internship Program**
> BharatCares & AICTE | Student: **Prashant**

---

## 📌 Project Overview

FinGuard AI is a production-ready, end-to-end credit risk analytics application built with Python and Streamlit. It combines classical machine learning, professional data visualisation, and IBM watsonx.ai generative AI to deliver a comprehensive loan default risk assessment platform.

**Domain:** Finance & Banking — Credit Risk & Loan Default Prediction

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **Auto Column Detection** | Detects all column names, types, target variable, and ID columns dynamically — no hardcoded column names |
| **Data Profiling** | Full schema inspection: dtypes, missing values, duplicates, cardinality, class imbalance |
| **Exploratory Data Analysis** | 8-panel EDA: distributions, correlations, box plots, outliers, scatter matrix |
| **7 ML Classifiers** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, KNN, Naive Bayes |
| **Model Evaluation** | ROC curves, PR curves, confusion matrices, feature importance, cross-validation |
| **Risk Predictor** | Real-time loan applicant risk scoring with 5-tier risk classification |
| **IBM watsonx.ai** | AI risk narratives, EDA insights & model commentary using IBM Granite |
| **Excel Export** | 7-sheet professional Excel report with styled tables and summaries |

---

## 📂 Project Structure

```
FinGuard-AI/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── utils/
    ├── __init__.py
    ├── data_loader.py        # Auto-detection engine & preprocessing
    ├── eda.py                # EDA charts and statistics
    ├── model_engine.py       # ML training, evaluation, prediction
    ├── watsonx_client.py     # IBM watsonx.ai integration
    └── report_generator.py   # Excel report generation
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit ≥ 1.35 | Web application framework |
| Pandas + NumPy | Data manipulation |
| Scikit-Learn | ML algorithms & evaluation |
| XGBoost | Gradient boosted trees |
| Plotly | Interactive visualisations |
| OpenPyXL | Excel report generation |
| IBM watsonx.ai | AI-powered insights (Granite LLM) |

---

## 📦 Installation

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd FinGuard-AI
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Dataset

**Dataset:** Bank Credit Default — Loan Default Prediction
**Source:** [Kaggle — kornilovag94](https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default)

**To use the dataset:**
1. Download the dataset from the Kaggle URL above
2. Upload the CSV file using the sidebar file uploader in the app

> ⚠️ **Important:** This application does **not** fabricate or generate any dataset.
> It works exclusively with the actual Kaggle dataset uploaded by the user.
> All column detection, feature engineering, and analysis are performed dynamically.

---

## 🤖 IBM watsonx.ai Setup (Optional)

To enable AI-powered risk narratives and insights:

1. Create an [IBM Cloud account](https://cloud.ibm.com)
2. Provision IBM watsonx.ai service
3. Navigate to IAM → API Keys → Create a new API key
4. Create or open a watsonx.ai project and copy the Project ID
5. Enter credentials in the FinGuard AI sidebar under **IBM watsonx.ai**

The app uses the **ibm/granite-13b-instruct-v2** foundation model.

---

## 🔧 Application Tabs

| Tab | Contents |
|---|---|
| 📊 Data Overview | Column classification, target analysis, summary stats, raw data preview |
| 🔍 EDA | 8-panel exploratory analysis with interactive Plotly charts |
| 🤖 Model Training | Pipeline description, training controls, metrics summary |
| 🏆 Model Evaluation | ROC curves, PR curves, confusion matrix, feature importance |
| 🔮 Risk Predictor | Dynamic input form, real-time prediction, AI narrative |
| 🤖 AI Insights | EDA insights + model insights via IBM Granite |
| 📥 Export Report | 7-sheet styled Excel report download |

---

## 🚦 Risk Tiers

| Probability | Tier |
|---|---|
| ≥ 75% | 🔴 Very High Risk |
| 55–75% | 🟠 High Risk |
| 35–55% | 🟡 Moderate Risk |
| 15–35% | 🟢 Low Risk |
| < 15% | ✅ Very Low Risk |

---

## 🎓 Academic Information

- **Program:** IBM SkillsBuild Data Analytics with AI
- **Internship Provider:** BharatCares
- **Approved By:** AICTE (All India Council for Technical Education)
- **Student:** Prashant
- **Project Title:** FinGuard AI — Bank Credit and Loan Default Risk Analytics Engine

---

## 📝 License

This project is developed for academic purposes as part of the IBM SkillsBuild internship program.
