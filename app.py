"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          FinGuard AI — Bank Credit & Loan Default Risk Analytics Engine      ║
║          IBM SkillsBuild · BharatCares · AICTE Academic Internship           ║
║          Student: Prashant                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Technology Stack: Python · Streamlit · Pandas · NumPy · Scikit-Learn
                  Plotly · OpenPyXL · IBM watsonx.ai (Granite)

Dataset: Bank Credit Default — Loan Default Prediction (Kaggle)
         https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default

IMPORTANT: This application performs auto-detection of all dataset columns.
           No column names are hardcoded. The app adapts to the actual dataset.
"""

from __future__ import annotations

import warnings
import os
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default",
        "About": (
            "FinGuard AI — Bank Credit & Loan Default Risk Analytics Engine\n\n"
            "IBM SkillsBuild Academic Internship | BharatCares & AICTE\n"
            "Student: Prashant"
        ),
    },
)

# ---------------------------------------------------------------------------
# Custom CSS — professional IBM-inspired dark/light theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Global Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Main background ─────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}

/* ── Header Banner ───────────────────────────────────────── */
.finguard-header {
    background: linear-gradient(135deg, #0f62fe 0%, #0043ce 50%, #001d6c 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.finguard-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.finguard-header p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
}

/* ── Metric Cards ────────────────────────────────────────── */
.metric-card {
    background: #f0f4ff;
    border: 1px solid #d0e0ff;
    border-left: 4px solid #0f62fe;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-card .label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #0043ce;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.2rem;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f62fe;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #697077;
    margin-top: 0.2rem;
}

/* ── Risk Badge ──────────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 24px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin: 0.5rem 0;
}

/* ── Section headings ────────────────────────────────────── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1f2328;
    border-bottom: 2px solid #0f62fe;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Info Boxes ──────────────────────────────────────────── */
.info-box {
    background: #edf5ff;
    border: 1px solid #97c1ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #1f2328;
    font-size: 0.9rem;
    line-height: 1.6;
}
.warn-box {
    background: #fff8e6;
    border: 1px solid #f1c21b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #4a3800;
    font-size: 0.9rem;
}
.success-box {
    background: #defbe6;
    border: 1px solid #42be65;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #044317;
    font-size: 0.9rem;
}
.error-box {
    background: #fff1f1;
    border: 1px solid #da1e28;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #750000;
    font-size: 0.9rem;
}

/* ── AI Response Box ─────────────────────────────────────── */
.ai-response {
    background: #f6f0ff;
    border: 1px solid #be95ff;
    border-left: 4px solid #8a3ffc;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    color: #1f2328;
    font-size: 0.9rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #001d6c;
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #a6c8ff !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Upload Button — dark theme ──────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #002080 !important;
    border: 1px solid #4589ff !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label {
    color: #a6c8ff !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #001450 !important;
    border: 2px dashed #4589ff !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #a6c8ff !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
    background: #0f62fe !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"]:hover {
    background: #0043ce !important;
}

/* ── Progress / spinner colour ───────────────────────────── */
.stProgress .st-bo { background-color: #0f62fe; }

/* ── Dataframe styling ───────────────────────────────────── */
.dataframe { font-size: 0.82rem !important; }

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f0f4ff;
    padding: 4px;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #0043ce;
    font-weight: 500;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: #0f62fe !important;
    color: white !important;
}

/* ── Footer ──────────────────────────────────────────────── */
.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #697077;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid #e5e7eb;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------

from utils.data_loader import load_dataset, profile_dataset, preprocess_for_model
from utils.eda import (
    compute_class_imbalance,
    compute_outlier_summary,
    compute_summary_stats,
    plot_boxplots_by_target,
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_numeric_distributions,
    plot_scatter_matrix,
    plot_target_distribution,
)
from utils.model_engine import (
    ModelResult,
    compute_risk_score,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_time,
    predict_single,
    select_best_model,
    train_models,
)
from utils.report_generator import generate_excel_report
from utils.watsonx_client import WatsonxClient


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "profile": None,
        "df": None,
        "model_results": None,
        "X_test": None,
        "y_test": None,
        "scaler": None,
        "feature_names": None,
        "target_classes": None,
        "best_model": None,
        "watsonx_client": None,
        "watsonx_connected": False,
        "current_tab": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.5rem;">🏦</span>
        <h2 style="margin:0; font-size:1.2rem; font-weight:700;">FinGuard AI</h2>
        <p style="font-size:0.72rem; opacity:0.7; margin:0;">Credit & Loan Default Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Dataset Upload ──────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader(
        "Upload Kaggle CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help=(
            "Upload the Bank Credit Default dataset from Kaggle:\n"
            "https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default"
        ),
    )

    st.markdown("---")

    # ── Model Settings ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model Settings")

    all_model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "K-Nearest Neighbours",
        "Naive Bayes",
    ]

    selected_models = st.multiselect(
        "Models to train",
        options=all_model_names,
        default=["Logistic Regression", "Random Forest", "XGBoost", "Gradient Boosting"],
        help="Select which classifiers to train and compare.",
    )

    test_size = st.slider(
        "Test split size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Fraction of data reserved for testing.",
    )

    cv_folds = st.slider(
        "Cross-validation folds",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of stratified CV folds for evaluation.",
    )

    best_metric = st.selectbox(
        "Best model selection metric",
        options=["roc_auc", "f1", "accuracy", "avg_precision"],
        index=0,
        help="Metric used to crown the best model.",
    )

    train_btn = st.button(
        "🚀 Train Models",
        use_container_width=True,
        disabled=(uploaded_file is None or not selected_models),
        type="primary",
    )

    st.markdown("---")

    # ── IBM watsonx.ai Settings ──────────────────────────────────────────────
    st.markdown("### 🤖 IBM watsonx.ai")
    st.markdown(
        '<p style="font-size:0.78rem; opacity:0.7;">Optional — AI-powered risk explanations</p>',
        unsafe_allow_html=True,
    )

    wx_api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter IBM Cloud API Key",
    )
    wx_project_id = st.text_input(
        "Project ID",
        placeholder="Enter watsonx.ai Project ID",
    )
    wx_url = st.selectbox(
        "Region URL",
        options=[
            "https://us-south.ml.cloud.ibm.com",
            "https://eu-de.ml.cloud.ibm.com",
            "https://eu-gb.ml.cloud.ibm.com",
            "https://jp-tok.ml.cloud.ibm.com",
            "https://au-syd.ml.cloud.ibm.com",
        ],
        help="Select the watsonx.ai region matching your IBM Cloud account.",
    )

    wx_connect_btn = st.button(
        "🔗 Connect to watsonx.ai",
        use_container_width=True,
        disabled=(not wx_api_key or not wx_project_id),
    )

    if wx_connect_btn and wx_api_key and wx_project_id:
        with st.spinner("Connecting to IBM watsonx.ai…"):
            client = WatsonxClient(api_key=wx_api_key, project_id=wx_project_id, url=wx_url)
            ok = client.connect()
            if ok:
                st.session_state.watsonx_client = client
                st.session_state.watsonx_connected = True
                st.success("✅ Connected!")
            else:
                st.error(f"Connection failed: {client.last_error}")
                st.session_state.watsonx_connected = False

    if st.session_state.watsonx_connected:
        st.markdown(
            '<div class="success-box" style="background:#defbe6;padding:0.5rem 0.8rem;'
            'border-radius:6px;font-size:0.8rem;">✅ watsonx.ai Active</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;opacity:0.5;text-align:center;">'
        'IBM SkillsBuild · BharatCares · AICTE<br>'
        'FinGuard AI v1.0.0'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------

st.markdown("""
<div class="finguard-header">
    <h1>🏦 FinGuard AI</h1>
    <p>Bank Credit &amp; Loan Default Risk Analytics Engine &nbsp;·&nbsp;
       IBM SkillsBuild Academic Internship &nbsp;·&nbsp;
       BharatCares &amp; AICTE &nbsp;·&nbsp;
       Student: Prashant</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Dataset Loading & Profiling
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    if st.session_state.df is None or st.session_state.get("_last_filename") != uploaded_file.name:
        with st.spinner(f"Loading and profiling **{uploaded_file.name}**…"):
            try:
                df = load_dataset(uploaded_file)
                profile = profile_dataset(df)
                st.session_state.df = df
                st.session_state.profile = profile
                st.session_state._last_filename = uploaded_file.name
                # Reset model state on new dataset
                st.session_state.model_results = None
                st.session_state.best_model = None
            except Exception as e:
                st.error(f"❌ Failed to load dataset: {e}")
                st.stop()

    profile = st.session_state.profile
    df = st.session_state.df

    # ── Quick KPI strip ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("📊 Total Records", f"{profile.n_rows:,}")
    with c2:
        st.metric("📋 Features", f"{len(profile.feature_cols)}")
    with c3:
        st.metric("🎯 Target", str(profile.target_col))
    with c4:
        miss_pct = round(profile.total_missing / max(profile.n_rows * profile.n_cols, 1) * 100, 1)
        st.metric("❓ Missing %", f"{miss_pct}%")
    with c5:
        st.metric("♻️ Duplicates", f"{profile.duplicate_rows:,}")
    with c6:
        st.metric("🤖 Models Ready", f"{len(selected_models)}")

    # Target warning if not detected
    if not profile.target_col:
        st.markdown(
            '<div class="warn-box">⚠️ No target column could be auto-detected. '
            'The application looked for columns with names like <code>default</code>, '
            '<code>status</code>, <code>label</code>, etc. '
            'Please ensure your dataset contains a clearly named target column.</div>',
            unsafe_allow_html=True,
        )

else:
    # ── Landing state (no file uploaded yet) ─────────────────────────────────
    st.markdown("""
    <div class="info-box">
        <h3 style="margin:0 0 0.8rem 0; color:#0f62fe;">👋 Welcome to FinGuard AI</h3>
        <p style="margin:0 0 0.5rem 0;">
            Upload the <strong>Bank Credit Default — Loan Default Prediction</strong> dataset from Kaggle
            using the sidebar to begin your analysis.
        </p>
        <p style="margin:0 0 0.5rem 0;">
            📥 Dataset URL:
            <a href="https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default" target="_blank">
            kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default</a>
        </p>
        <ul style="margin: 0.5rem 0 0 1rem; padding:0;">
            <li>The app will <strong>auto-detect</strong> all columns, data types, and the target variable</li>
            <li>No column names are hardcoded — the app adapts to the actual dataset</li>
            <li>Supports CSV and Excel formats</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Feature overview cards
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">📊 Exploratory Analysis</div>
            <div style="font-size:0.88rem; color:#1f2328; margin-top:0.4rem;">
                Auto-profiling · Distributions · Correlations · Outliers · Class Balance
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">🤖 Machine Learning</div>
            <div style="font-size:0.88rem; color:#1f2328; margin-top:0.4rem;">
                7 Classifiers · ROC-AUC · Confusion Matrix · Feature Importance · CV
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">🔮 Risk Prediction</div>
            <div style="font-size:0.88rem; color:#1f2328; margin-top:0.4rem;">
                Real-time scoring · 5 risk tiers · IBM watsonx.ai AI narrative
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

if train_btn and profile.target_col:
    with st.spinner("⚙️ Preprocessing data and training models… this may take a minute…"):
        try:
            X, y, feature_names, target_classes = preprocess_for_model(df, profile)

            results, X_test, y_test, scaler = train_models(
                X, y, feature_names,
                test_size=test_size,
                cv_folds=cv_folds,
                scale_features=True,
                selected_models=selected_models if selected_models else None,
            )

            best_model = select_best_model(results, metric=best_metric)

            st.session_state.model_results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.feature_names = feature_names
            st.session_state.target_classes = target_classes
            st.session_state.best_model = best_model

            st.success(
                f"✅ Training complete! Best model: **{best_model.name}** "
                f"(ROC-AUC: {best_model.roc_auc:.4f}, F1: {best_model.f1:.4f})"
            )

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.exception(e)


# ---------------------------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------------------------

TAB_LABELS = [
    "📊 Data Overview",
    "🔍 EDA",
    "🤖 Model Training",
    "🏆 Model Evaluation",
    "🔮 Risk Predictor",
    "🤖 AI Insights",
    "📥 Export Report",
]

tabs = st.tabs(TAB_LABELS)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown('<div class="section-title">Dataset Auto-Detection Report</div>', unsafe_allow_html=True)

    # ── Detection Results ────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("#### 📋 Column Classification")

        col_data = []
        for col in profile.all_columns:
            col_type = (
                "🎯 Target" if col == profile.target_col else
                "🔑 Identifier" if col in profile.id_cols else
                "📅 Datetime" if col in profile.datetime_cols else
                "🔢 Numeric" if col in profile.numeric_cols else
                "🔤 Categorical"
            )
            flag = ""
            if col in profile.binary_cols:
                flag = "⊙ Binary"
            if col in profile.high_cardinality_cols:
                flag = "⚡ High Cardinality"
            col_data.append({
                "Column": col,
                "Detected Type": col_type,
                "Data Type": str(df[col].dtype),
                "Unique Values": df[col].nunique(),
                "Missing": f"{profile.missing_pct.get(col, 0):.1f}%",
                "Notes": flag,
            })

        st.dataframe(
            pd.DataFrame(col_data),
            use_container_width=True,
            height=420,
        )

    with col_r:
        st.markdown("#### 🎯 Target Variable Analysis")

        if profile.target_col:
            # Target info
            dist = profile.target_distribution
            total = sum(dist.values())
            for cls, cnt in dist.items():
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Class: {cls}</div>'
                    f'<div class="value">{cnt:,}</div>'
                    f'<div class="sub">{cnt/total*100:.1f}% of dataset</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            imb = compute_class_imbalance(profile)
            if imb:
                sev_colour = (
                    "#da1e28" if imb["severity"] == "Severe" else
                    "#ff832b" if imb["severity"] == "Moderate" else
                    "#24a148"
                )
                st.markdown(
                    f'<div class="warn-box" style="margin-top:0.5rem;">'
                    f'<strong>Imbalance Ratio:</strong> {imb["imbalance_ratio"]}:1<br>'
                    f'<strong>Severity:</strong> <span style="color:{sev_colour};font-weight:600;">'
                    f'{imb["severity"]}</span><br>'
                    f'<strong>Recommendation:</strong> {imb["recommendation"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No target column detected.")

        # Derived features
        if profile.derived_cols:
            st.markdown("#### 🔧 Derived Features Available")
            for feat, desc in profile.derived_cols.items():
                st.markdown(
                    f'<div class="info-box" style="margin-bottom:0.4rem;">'
                    f'<strong>{feat}</strong><br>'
                    f'<span style="font-size:0.82rem;">{desc}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Summary Statistics Table ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Summary Statistics — Numeric Features</div>', unsafe_allow_html=True)
    stats = compute_summary_stats(profile)
    if not stats.empty:
        st.dataframe(stats.style.format("{:.4f}", subset=pd.IndexSlice[:, stats.select_dtypes(include=np.number).columns]), use_container_width=True, height=300)
    else:
        st.info("No numeric features found.")

    st.markdown("---")

    # ── Raw Data Preview ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Raw Data Preview (First 100 Rows)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True, height=320)

    col_a, col_b = st.columns(2)
    with col_a:
        if profile.id_cols:
            st.markdown(
                f'<div class="info-box">🔑 <strong>Identifier columns detected:</strong> '
                f'{", ".join(profile.id_cols)} — excluded from model training.</div>',
                unsafe_allow_html=True,
            )
    with col_b:
        if profile.high_cardinality_cols:
            st.markdown(
                f'<div class="warn-box">⚡ <strong>High-cardinality columns</strong> '
                f'({", ".join(profile.high_cardinality_cols)}) will be dropped before training '
                f'to prevent memory issues.</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    eda_tabs = st.tabs([
        "🎯 Target Distribution",
        "📈 Numeric Dist.",
        "🔤 Categorical Dist.",
        "🔥 Correlations",
        "📦 Box Plots",
        "🔍 Scatter Matrix",
        "❓ Missing Values",
        "⚠️ Outliers",
    ])

    with eda_tabs[0]:
        fig = plot_target_distribution(profile)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target column not detected.")

    with eda_tabs[1]:
        if profile.numeric_features:
            n_show = st.slider("Max features to plot", 3, min(20, len(profile.numeric_features)),
                               min(12, len(profile.numeric_features)), key="hist_n")
            fig = plot_numeric_distributions(profile, max_cols=n_show)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric features found in the dataset.")

    with eda_tabs[2]:
        if profile.categorical_features:
            fig = plot_categorical_distributions(profile, max_cols=8)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if profile.high_cardinality_cols:
                st.markdown(
                    f'<div class="warn-box">⚡ High-cardinality columns not plotted: '
                    f'{", ".join(profile.high_cardinality_cols)}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No categorical features found in the dataset.")

    with eda_tabs[3]:
        if len(profile.numeric_features) >= 2:
            fig = plot_correlation_heatmap(profile)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Top correlations with target (if numeric target)
            if profile.target_col and profile.target_col in profile.numeric_cols:
                corr_with_target = (
                    df[profile.numeric_features + [profile.target_col]]
                    .corr()[profile.target_col]
                    .drop(profile.target_col)
                    .sort_values(key=abs, ascending=False)
                )
                st.markdown("##### Top Correlations with Target")
                st.dataframe(
                    corr_with_target.head(20).reset_index().rename(
                        columns={"index": "Feature", profile.target_col: "Correlation"}
                    ),
                    use_container_width=True,
                )
        else:
            st.info("At least 2 numeric features are required for correlation analysis.")

    with eda_tabs[4]:
        if profile.numeric_features and profile.target_col:
            n_box = st.slider("Max features to plot", 2, min(16, len(profile.numeric_features)),
                              min(8, len(profile.numeric_features)), key="box_n")
            fig = plot_boxplots_by_target(profile, max_cols=n_box)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Numeric features and target column are required for box plots.")

    with eda_tabs[5]:
        if len(profile.numeric_features) >= 2 and profile.target_col:
            fig = plot_scatter_matrix(profile)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.caption("Showing top 5 numeric features. Coloured by target class.")
        else:
            st.info("At least 2 numeric features and a target column are required.")

    with eda_tabs[6]:
        miss_fig = plot_missing_values(profile)
        if miss_fig:
            st.plotly_chart(miss_fig, use_container_width=True)
            st.dataframe(
                pd.DataFrame({
                    "Column": list(profile.missing_counts.keys()),
                    "Missing Count": list(profile.missing_counts.values()),
                    "Missing %": list(profile.missing_pct.values()),
                }).sort_values("Missing %", ascending=False),
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="success-box">✅ No missing values detected in the dataset!</div>',
                unsafe_allow_html=True,
            )

    with eda_tabs[7]:
        if profile.numeric_features:
            outlier_df = compute_outlier_summary(profile)
            st.dataframe(
                outlier_df.style.background_gradient(
                    subset=["Outlier %"],
                    cmap="RdYlGn_r",
                ),
                use_container_width=True,
                height=400,
            )
            total_outliers = outlier_df["Outlier Count"].sum()
            st.caption(
                f"IQR method (1.5×IQR fence). Total outlier instances across all features: {total_outliers:,}"
            )
        else:
            st.info("No numeric features for outlier analysis.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="section-title">Machine Learning — Model Training</div>', unsafe_allow_html=True)

    if not profile.target_col:
        st.markdown(
            '<div class="error-box">❌ Cannot train models: No target column detected.</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.model_results is None:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin:0 0 0.5rem 0;">🚀 Ready to Train</h4>
            <ol style="margin:0; padding-left:1.2rem;">
                <li>Select models in the sidebar (2–4 recommended for first run)</li>
                <li>Adjust test split size and CV folds</li>
                <li>Click <strong>🚀 Train Models</strong> in the sidebar</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Training pipeline explanation
        st.markdown("---")
        st.markdown("#### Training Pipeline")
        pipeline_cols = st.columns(5)
        steps = [
            ("1️⃣", "Auto\nPreprocess", "Encodes categoricals, fills NaNs, scales features"),
            ("2️⃣", "Train/Test\nSplit", f"Stratified split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test"),
            ("3️⃣", "Train\nModels", f"Fits {len(selected_models)} selected classifiers"),
            ("4️⃣", "Cross\nValidation", f"{cv_folds}-fold stratified CV on training set"),
            ("5️⃣", "Evaluate\n& Compare", "ROC-AUC, F1, Precision, Recall, Confusion Matrix"),
        ]
        for col, (icon, title, desc) in zip(pipeline_cols, steps):
            with col:
                st.markdown(
                    f'<div class="metric-card" style="text-align:center;">'
                    f'<div style="font-size:1.5rem;">{icon}</div>'
                    f'<div class="label" style="margin:0.3rem 0;">{title}</div>'
                    f'<div style="font-size:0.78rem;color:#697077;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        results = st.session_state.model_results
        best = st.session_state.best_model
        feature_names = st.session_state.feature_names

        # ── Best Model Banner ────────────────────────────────────────────────
        st.markdown(
            f'<div class="success-box">'
            f'🏆 <strong>Best Model:</strong> {best.name} &nbsp;|&nbsp; '
            f'ROC-AUC: <strong>{best.roc_auc:.4f}</strong> &nbsp;|&nbsp; '
            f'F1: <strong>{best.f1:.4f}</strong> &nbsp;|&nbsp; '
            f'Accuracy: <strong>{best.accuracy:.4f}</strong> &nbsp;|&nbsp; '
            f'Train Time: {best.train_time:.3f}s'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Metrics Table ────────────────────────────────────────────────────
        st.markdown("#### 📊 Model Metrics Summary")
        summary_rows = []
        for name, res in results.items():
            summary_rows.append({
                "Model": f"{'🏆 ' if name == best.name else ''}{name}",
                "Accuracy": f"{res.accuracy:.4f}",
                "Precision": f"{res.precision:.4f}",
                "Recall": f"{res.recall:.4f}",
                "F1-Score": f"{res.f1:.4f}",
                "ROC-AUC": f"{res.roc_auc:.4f}",
                "Avg Precision": f"{res.avg_precision:.4f}",
                "CV Mean AUC": f"{res.cv_mean:.4f} ± {res.cv_std:.4f}",
                "Train Time": f"{res.train_time:.3f}s",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📈 Visual Comparison")
        fig_comp = plot_model_comparison(results)
        st.plotly_chart(fig_comp, use_container_width=True)

        col_cv, col_time = st.columns(2)
        with col_cv:
            st.plotly_chart(plot_cv_scores(results), use_container_width=True)
        with col_time:
            st.plotly_chart(plot_training_time(results), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown('<div class="section-title">Detailed Model Evaluation</div>', unsafe_allow_html=True)

    if st.session_state.model_results is None:
        st.info("Train models first using the sidebar.")
    else:
        results = st.session_state.model_results
        y_test = st.session_state.y_test
        target_classes = st.session_state.target_classes
        feature_names = st.session_state.feature_names
        best = st.session_state.best_model

        # Model selector
        selected_eval_model = st.selectbox(
            "Inspect model:",
            options=list(results.keys()),
            index=list(results.keys()).index(best.name),
        )
        res = results[selected_eval_model]

        eval_tabs = st.tabs([
            "📊 ROC Curves",
            "📉 Precision-Recall",
            "🔲 Confusion Matrix",
            "⭐ Feature Importance",
            "📋 Classification Report",
        ])

        with eval_tabs[0]:
            if profile.is_binary_target:
                fig = plot_roc_curves(results, y_test)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROC curves are only available for binary classification tasks.")

        with eval_tabs[1]:
            if profile.is_binary_target:
                fig = plot_precision_recall_curves(results, y_test)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Precision-Recall curves are only available for binary classification.")

        with eval_tabs[2]:
            fig_cm = plot_confusion_matrix(res, target_classes)
            if fig_cm:
                col_cm, col_cm_stats = st.columns([1, 1])
                with col_cm:
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col_cm_stats:
                    if res.confusion is not None and len(target_classes) == 2:
                        cm = res.confusion
                        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                        st.markdown("##### Confusion Matrix Breakdown")
                        st.markdown(
                            f'<div class="info-box">'
                            f'<strong>True Negatives (TN):</strong> {tn:,}<br>'
                            f'<strong>False Positives (FP):</strong> {fp:,}<br>'
                            f'<strong>False Negatives (FN):</strong> {fn:,}<br>'
                            f'<strong>True Positives (TP):</strong> {tp:,}<br><br>'
                            f'<strong>Sensitivity / Recall:</strong> {tp/(tp+fn+1e-9):.4f}<br>'
                            f'<strong>Specificity:</strong> {tn/(tn+fp+1e-9):.4f}<br>'
                            f'<strong>Precision (PPV):</strong> {tp/(tp+fp+1e-9):.4f}<br>'
                            f'<strong>NPV:</strong> {tn/(tn+fn+1e-9):.4f}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        with eval_tabs[3]:
            fig_fi = plot_feature_importance(res, feature_names, top_n=25)
            if fig_fi:
                top_n_fi = st.slider("Top N features", 5, min(50, len(feature_names)), 20, key="fi_slider")
                fig_fi = plot_feature_importance(res, feature_names, top_n=top_n_fi)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info(f"{res.name} does not support feature importance extraction.")

        with eval_tabs[4]:
            from sklearn.metrics import classification_report
            cr = classification_report(
                y_test, res.y_pred,
                target_names=[str(c) for c in target_classes],
                output_dict=True, zero_division=0
            )
            cr_df = pd.DataFrame(cr).T
            st.dataframe(
                cr_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None),
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK PREDICTOR
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown('<div class="section-title">🔮 Real-Time Loan Default Risk Predictor</div>', unsafe_allow_html=True)

    if st.session_state.model_results is None:
        st.info("Train models first using the sidebar before using the predictor.")
    else:
        results = st.session_state.model_results
        feature_names = st.session_state.feature_names
        scaler = st.session_state.scaler
        target_classes = st.session_state.target_classes
        best = st.session_state.best_model

        # Model selector for prediction
        pred_model_name = st.selectbox(
            "Select model for prediction:",
            options=list(results.keys()),
            index=list(results.keys()).index(best.name),
            key="pred_model_sel",
        )
        pred_model = results[pred_model_name].model

        st.markdown(
            '<div class="info-box">ℹ️ Input values for each feature below. '
            'The form is auto-generated from the <strong>actual training features</strong> '
            'detected in your dataset. Categorical features that were one-hot encoded '
            'are shown as numeric toggle inputs (0 = absent, 1 = present).</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Dynamically generate input form from actual feature names
        input_dict: dict = {}

        # Group features into columns of 3
        cols_per_row = 3
        feat_chunks = [feature_names[i:i+cols_per_row] for i in range(0, len(feature_names), cols_per_row)]

        with st.form("risk_predictor_form"):
            for chunk in feat_chunks:
                form_cols = st.columns(cols_per_row)
                for i, feat in enumerate(chunk):
                    with form_cols[i]:
                        # Check if feature is binary (OHE indicator)
                        if feat.endswith("_1") or feat.endswith("_True") or feat.endswith("_Yes"):
                            val = st.selectbox(feat, options=[0, 1], key=f"inp_{feat}")
                        else:
                            # Use actual column stats as defaults
                            orig_col = feat.split("_")[0] if "_" in feat else feat
                            if orig_col in df.columns and pd.api.types.is_numeric_dtype(df[orig_col]):
                                med = float(df[orig_col].median())
                                mn = float(df[orig_col].min())
                                mx = float(df[orig_col].max())
                                val = st.number_input(
                                    feat,
                                    min_value=mn,
                                    max_value=mx,
                                    value=med,
                                    step=(mx - mn) / 100 if mx != mn else 1.0,
                                    key=f"inp_{feat}",
                                    format="%.4f",
                                )
                            else:
                                val = st.number_input(feat, value=0.0, key=f"inp_{feat}")
                        input_dict[feat] = val

            predict_btn = st.form_submit_button("🔮 Predict Default Risk", type="primary", use_container_width=True)

        if predict_btn:
            with st.spinner("Computing risk score…"):
                pred_class, proba, risk_label, risk_colour = predict_single(
                    pred_model, scaler, input_dict, feature_names
                )

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            r1, r2, r3 = st.columns(3)
            with r1:
                outcome = "DEFAULT" if pred_class == 1 else "NON-DEFAULT"
                outcome_colour = "#da1e28" if pred_class == 1 else "#24a148"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Prediction</div>'
                    f'<div class="value" style="color:{outcome_colour};">{outcome}</div>'
                    f'<div class="sub">Class: {target_classes[pred_class] if pred_class < len(target_classes) else pred_class}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Default Probability</div>'
                    f'<div class="value" style="color:{risk_colour};">{proba:.1%}</div>'
                    f'<div class="sub">Model: {pred_model_name}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">Risk Tier</div>'
                    f'<div class="value" style="font-size:1rem; color:{risk_colour};">{risk_label}</div>'
                    f'<div class="sub">Based on probability threshold</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Risk gauge bar
            gauge_pct = int(proba * 100)
            gauge_colour = risk_colour.lstrip("#")
            st.markdown(
                f'<div style="margin:1rem 0 0.3rem 0; font-size:0.85rem; font-weight:600; color:#1f2328;">Default Probability Gauge</div>'
                f'<div style="background:#e5e7eb; border-radius:99px; height:18px; overflow:hidden;">'
                f'<div style="width:{gauge_pct}%; background:#{gauge_colour}; height:100%; '
                f'border-radius:99px; transition:width 0.4s;"></div>'
                f'</div>'
                f'<div style="font-size:0.78rem; color:#697077; margin-top:0.2rem;">{gauge_pct}%</div>',
                unsafe_allow_html=True,
            )

            # ── watsonx.ai AI Explanation ────────────────────────────────────
            if st.session_state.watsonx_connected:
                st.markdown("---")
                st.markdown("### 🤖 IBM watsonx.ai — AI Risk Narrative")

                with st.spinner("Generating AI risk explanation with IBM Granite…"):
                    # Get feature importances for top features
                    best_res = results[pred_model_name]
                    top_features = []
                    if best_res.feature_importances is not None:
                        fi = best_res.feature_importances
                        idx = np.argsort(fi)[::-1][:8]
                        top_features = [
                            (feature_names[i], float(input_dict.get(feature_names[i], 0)))
                            for i in idx
                        ]

                    try:
                        wx = st.session_state.watsonx_client
                        explanation = wx.explain_risk(
                            applicant_data=input_dict,
                            predicted_class=pred_class,
                            probability=proba,
                            risk_label=risk_label,
                            top_features=top_features,
                            dataset_name=uploaded_file.name if uploaded_file else "credit dataset",
                        )
                        st.markdown(
                            f'<div class="ai-response">{explanation}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"AI explanation failed: {e}")
            else:
                st.markdown(
                    '<div class="warn-box">💡 Connect IBM watsonx.ai in the sidebar to '
                    'generate AI-powered risk narratives for this prediction.</div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown('<div class="section-title">🤖 IBM watsonx.ai — AI-Powered Insights</div>', unsafe_allow_html=True)

    if not st.session_state.watsonx_connected:
        st.markdown("""
        <div class="warn-box">
            <h4 style="margin:0 0 0.5rem 0;">🤖 IBM watsonx.ai Not Connected</h4>
            <p style="margin:0;">Connect your IBM watsonx.ai credentials in the sidebar to unlock AI-powered insights powered by IBM Granite foundation models.</p>
            <br>
            <p style="margin:0; font-size:0.85rem;">You need:</p>
            <ul style="margin:0.3rem 0 0 1rem;">
                <li>IBM Cloud account with watsonx.ai access</li>
                <li>API Key from IBM Cloud IAM</li>
                <li>watsonx.ai Project ID</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        wx = st.session_state.watsonx_client

        ai_tabs = st.tabs(["📊 EDA Insights", "🤖 Model Insights"])

        with ai_tabs[0]:
            st.markdown("#### AI Analysis of Dataset Characteristics")
            if st.button("🔍 Generate EDA Insights", type="primary"):
                with st.spinner("IBM Granite is analysing your dataset…"):
                    try:
                        # Compute top correlations
                        top_corrs = []
                        if len(profile.numeric_features) >= 2:
                            corr_mat = df[profile.numeric_features].corr(numeric_only=True)
                            corr_pairs = []
                            cols_n = corr_mat.columns.tolist()
                            for i in range(len(cols_n)):
                                for j in range(i + 1, len(cols_n)):
                                    corr_pairs.append((cols_n[i], cols_n[j], corr_mat.iloc[i, j]))
                            top_corrs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]

                        imb = compute_class_imbalance(profile)
                        insight = wx.generate_eda_insights(
                            dataset_summary={
                                "rows": profile.n_rows,
                                "cols": profile.n_cols,
                                "missing_pct": round(
                                    profile.total_missing / max(profile.n_rows * profile.n_cols, 1) * 100, 2
                                ),
                                "duplicates": profile.duplicate_rows,
                            },
                            target_col=str(profile.target_col),
                            imbalance_info=imb,
                            top_correlations=top_corrs,
                        )
                        st.markdown(
                            f'<div class="ai-response">{insight}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"AI insight generation failed: {e}")

        with ai_tabs[1]:
            st.markdown("#### AI Analysis of Model Performance")
            if st.session_state.model_results is None:
                st.info("Train models first to generate model insights.")
            elif st.button("🤖 Generate Model Insights", type="primary"):
                with st.spinner("IBM Granite is analysing model results…"):
                    try:
                        results = st.session_state.model_results
                        best = st.session_state.best_model
                        imb = compute_class_imbalance(profile)
                        insight = wx.generate_model_insights(
                            best_model_name=best.name,
                            best_roc_auc=best.roc_auc,
                            best_f1=best.f1,
                            all_model_scores=[
                                {
                                    "name": n,
                                    "roc_auc": r.roc_auc,
                                    "f1": r.f1,
                                    "accuracy": r.accuracy,
                                }
                                for n, r in results.items()
                            ],
                            dataset_rows=profile.n_rows,
                            is_imbalanced=bool(imb and imb.get("imbalance_ratio", 1) > 3),
                        )
                        st.markdown(
                            f'<div class="ai-response">{insight}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"AI insight generation failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPORT REPORT
# ════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown('<div class="section-title">📥 Export Professional Report</div>', unsafe_allow_html=True)

    if st.session_state.model_results is None:
        st.info("Train models first to generate the export report.")
    else:
        results = st.session_state.model_results
        best = st.session_state.best_model
        feature_names = st.session_state.feature_names

        st.markdown("""
        <div class="info-box">
            <h4 style="margin:0 0 0.5rem 0;">📊 Excel Report Contents</h4>
            <ul style="margin:0; padding-left:1.2rem;">
                <li><strong>Executive Summary</strong> — Project overview & best model stats</li>
                <li><strong>Dataset Overview</strong> — Auto-detected schema, shape, types</li>
                <li><strong>Data Profile</strong> — Column-level statistics</li>
                <li><strong>Class Distribution</strong> — Target variable breakdown</li>
                <li><strong>Missing Values</strong> — Missing value analysis per column</li>
                <li><strong>Model Results</strong> — All models, all metrics</li>
                <li><strong>Feature Importance</strong> — Top features from best model</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("📥 Generate & Download Excel Report", type="primary", use_container_width=True):
            with st.spinner("Generating multi-sheet Excel report…"):
                try:
                    report_bytes = generate_excel_report(
                        profile=profile,
                        results=results,
                        best_model=best,
                        feature_names=feature_names,
                    )
                    from datetime import datetime
                    filename = f"FinGuard_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.download_button(
                        label="💾 Download Report",
                        data=report_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                    st.success(f"✅ Report generated: {filename}")
                except ImportError:
                    st.error("openpyxl is not installed. Run: pip install openpyxl")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
                    st.exception(e)

        st.markdown("---")
        st.markdown("#### 📋 Current Session Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">Dataset</div>'
                f'<div class="value" style="font-size:1.1rem;">{uploaded_file.name if uploaded_file else "—"}</div>'
                f'<div class="sub">{profile.n_rows:,} rows · {profile.n_cols} columns</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">Target Variable</div>'
                f'<div class="value" style="font-size:1.1rem;">{profile.target_col}</div>'
                f'<div class="sub">Classes: {profile.target_classes}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with summary_col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">Best Model</div>'
                f'<div class="value" style="font-size:1.1rem;">{best.name}</div>'
                f'<div class="sub">ROC-AUC: {best.roc_auc:.4f} · F1: {best.f1:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">Models Evaluated</div>'
                f'<div class="value" style="font-size:1.1rem;">{len(results)}</div>'
                f'<div class="sub">{", ".join(results.keys())}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<div class="footer">
    🏦 <strong>FinGuard AI</strong> — Bank Credit &amp; Loan Default Risk Analytics Engine<br>
    IBM SkillsBuild Data Analytics with AI &nbsp;·&nbsp; Academic Internship Program<br>
    BharatCares &amp; AICTE &nbsp;·&nbsp; Student: <strong>Prashant</strong> &nbsp;·&nbsp;
    Powered by IBM watsonx.ai (Granite) · Scikit-Learn · Streamlit
</div>
""", unsafe_allow_html=True)
