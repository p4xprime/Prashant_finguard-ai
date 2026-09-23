"""
FinGuard AI — ML Model Engine
Trains multiple classifiers, evaluates them, and provides prediction
capabilities. All feature handling is driven by actual data profile.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

IBM_BLUE = "#0f62fe"
IBM_RED = "#da1e28"
IBM_GREEN = "#24a148"
IBM_ORANGE = "#ff832b"
IBM_PURPLE = "#8a3ffc"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#1f2328"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def _get_model_registry(class_weight: str = "balanced") -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight=class_weight, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight=class_weight,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        ),
        **({"XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=1,
        )} if _XGB_AVAILABLE else {}),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }


# ---------------------------------------------------------------------------
# Model Results Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    model: Any
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float
    train_time: float
    cv_mean: float
    cv_std: float
    y_pred: np.ndarray = field(default_factory=np.array)
    y_proba: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    feature_importances: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------

def train_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
    scale_features: bool = True,
    selected_models: Optional[list[str]] = None,
) -> tuple[dict[str, ModelResult], np.ndarray, np.ndarray, StandardScaler]:
    """
    Train multiple models and return results dict + test split.
    Returns: (results, X_test, y_test, scaler)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    if scale_features:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Imbalance ratio for XGBoost
    classes = np.unique(y_train)
    if len(classes) == 2:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        spw = n_neg / max(n_pos, 1)
    else:
        spw = 1

    registry = _get_model_registry()
    if "XGBoost" in registry:
        registry["XGBoost"].set_params(scale_pos_weight=spw)

    if selected_models:
        registry = {k: v for k, v in registry.items() if k in selected_models}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results: dict[str, ModelResult] = {}

    for name, model in registry.items():
        t0 = time.time()
        model.fit(X_train, y_train)

        elapsed = time.time() - t0

        y_pred = model.predict(X_test)

        # Probabilities (if available)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X_test)
            y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            y_proba = None

        roc = (
            roc_auc_score(y_test, y_proba)
            if y_proba is not None and len(np.unique(y_test)) == 2
            else 0.5
        )

        avg_prec = (
            average_precision_score(y_test, y_proba)
            if y_proba is not None and len(np.unique(y_test)) == 2
            else 0.0
        )

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )

        # Feature importances
        fi = None
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "coef_"):
            fi = np.abs(model.coef_[0])

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        # Use weighted average for multi-class safety
        precision = report.get("weighted avg", {}).get("precision", 0)
        recall = report.get("weighted avg", {}).get("recall", 0)
        f1 = report.get("weighted avg", {}).get("f1-score", 0)

        results[name] = ModelResult(
            name=name,
            model=model,
            accuracy=float(np.mean(y_pred == y_test)),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc),
            avg_precision=float(avg_prec),
            train_time=float(elapsed),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            y_pred=y_pred,
            y_proba=y_proba,
            confusion=confusion_matrix(y_test, y_pred),
            feature_importances=fi,
        )

    return results, X_test, y_test, scaler


# ---------------------------------------------------------------------------
# Best Model Selector
# ---------------------------------------------------------------------------

def select_best_model(results: dict[str, ModelResult], metric: str = "roc_auc") -> ModelResult:
    """Return the ModelResult with the highest value of the chosen metric."""
    return max(results.values(), key=lambda r: getattr(r, metric, 0))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _layout(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=16)))
    return fig


def plot_model_comparison(results: dict[str, ModelResult]):
    """Grouped bar chart comparing all metrics across models."""
    names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colours = [IBM_BLUE, IBM_GREEN, IBM_ORANGE, IBM_PURPLE, IBM_RED]

    fig = go.Figure()
    for metric, colour in zip(metrics, colours):
        values = [round(getattr(results[n], metric) * 100, 2) for n in names]
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=names,
            y=values,
            marker_color=colour,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        ))

    _layout(fig, "Model Comparison — All Metrics (%)")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 110]))
    return fig


def plot_roc_curves(results: dict[str, ModelResult], y_test: np.ndarray):
    """ROC curves for all models (binary classification)."""
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dot", color="grey", width=1))

    palette = [IBM_BLUE, IBM_RED, IBM_GREEN, IBM_PURPLE, IBM_ORANGE, IBM_CYAN, IBM_TEAL] \
        if False else ["#0f62fe", "#da1e28", "#24a148", "#8a3ffc", "#ff832b", "#1192e8", "#009d9a"]

    for i, (name, res) in enumerate(results.items()):
        if res.y_proba is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, res.y_proba)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{name} (AUC={res.roc_auc:.3f})",
                line=dict(color=palette[i % len(palette)], width=2),
            ))

    _layout(fig, "ROC Curves — All Models")
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    fig.update_layout(legend=dict(x=0.6, y=0.1))
    return fig


def plot_precision_recall_curves(results: dict[str, ModelResult], y_test: np.ndarray):
    """Precision-Recall curves for all models."""
    fig = go.Figure()
    palette = ["#0f62fe", "#da1e28", "#24a148", "#8a3ffc", "#ff832b", "#1192e8", "#009d9a"]

    for i, (name, res) in enumerate(results.items()):
        if res.y_proba is not None and len(np.unique(y_test)) == 2:
            prec, rec, _ = precision_recall_curve(y_test, res.y_proba)
            pr_auc = auc(rec, prec)
            fig.add_trace(go.Scatter(
                x=rec, y=prec,
                mode="lines",
                name=f"{name} (AP={pr_auc:.3f})",
                line=dict(color=palette[i % len(palette)], width=2),
            ))

    _layout(fig, "Precision-Recall Curves — All Models")
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    fig.update_layout(legend=dict(x=0.01, y=0.01))
    return fig


def plot_confusion_matrix(result: ModelResult, target_classes: list):
    """Heatmap confusion matrix for a single model."""
    cm = result.confusion
    if cm is None:
        return None

    labels = [str(c) for c in target_classes]
    text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        colorscale=[[0, "#ffffff"], [1, IBM_BLUE]],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=18),
        showscale=False,
    ))
    _layout(fig, f"Confusion Matrix — {result.name}")
    fig.update_layout(height=350)
    return fig


def plot_feature_importance(result: ModelResult, feature_names: list[str], top_n: int = 20):
    """Horizontal bar chart of top-N feature importances."""
    if result.feature_importances is None:
        return None

    fi = result.feature_importances
    names = np.array(feature_names)

    idx = np.argsort(fi)[-top_n:]
    fig = go.Figure(go.Bar(
        x=fi[idx],
        y=names[idx],
        orientation="h",
        marker=dict(
            color=fi[idx],
            colorscale=[[0, "#e5e7eb"], [1, IBM_BLUE]],
        ),
    ))
    _layout(fig, f"Top {top_n} Feature Importances — {result.name}")
    fig.update_layout(height=max(300, top_n * 22))
    return fig


def plot_training_time(results: dict[str, ModelResult]):
    """Bar chart of training times."""
    names = list(results.keys())
    times = [round(results[n].train_time, 3) for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=times,
        marker_color=IBM_PURPLE,
        text=[f"{t:.3f}s" for t in times],
        textposition="outside",
    ))
    _layout(fig, "Model Training Time (seconds)")
    fig.update_layout(yaxis=dict(title="Seconds"))
    return fig


# ---------------------------------------------------------------------------
# CV Score Plot
# ---------------------------------------------------------------------------

def plot_cv_scores(results: dict[str, ModelResult]):
    """Error bar chart of cross-validation ROC-AUC scores."""
    names = list(results.keys())
    means = [results[n].cv_mean for n in names]
    stds = [results[n].cv_std for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=[m * 100 for m in means],
        error_y=dict(type="data", array=[s * 100 for s in stds], visible=True),
        marker_color=IBM_TEAL if False else "#009d9a",
        text=[f"{m*100:.1f}%" for m in means],
        textposition="outside",
    ))
    _layout(fig, f"Cross-Validation ROC-AUC (mean ± std, {5}-fold)")
    fig.update_layout(yaxis=dict(title="ROC-AUC %", range=[0, 110]))
    return fig


IBM_TEAL = "#009d9a"
IBM_CYAN = "#1192e8"


# ---------------------------------------------------------------------------
# Risk Scoring
# ---------------------------------------------------------------------------

def compute_risk_score(proba: float) -> tuple[str, str]:
    """Convert default probability to risk tier and colour."""
    if proba >= 0.75:
        return "🔴 Very High Risk", "#da1e28"
    elif proba >= 0.55:
        return "🟠 High Risk", "#ff832b"
    elif proba >= 0.35:
        return "🟡 Moderate Risk", "#f1c21b"
    elif proba >= 0.15:
        return "🟢 Low Risk", "#24a148"
    else:
        return "✅ Very Low Risk", "#198038"


def predict_single(
    model: Any,
    scaler: StandardScaler,
    input_dict: dict,
    feature_names: list[str],
) -> tuple[int, float, str, str]:
    """
    Predict default risk for a single applicant.
    Returns (predicted_class, probability, risk_label, colour).
    """
    row = pd.DataFrame([input_dict])

    # Align to training feature order
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]
    row = row.fillna(0)

    X = scaler.transform(row.values.astype(float))

    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    else:
        proba = float(pred)

    label, colour = compute_risk_score(proba)
    return pred, proba, label, colour
