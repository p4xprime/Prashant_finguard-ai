"""
FinGuard AI — EDA Module
All charts and statistical summaries are driven entirely by the
actual DataProfile — no hardcoded column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from utils.data_loader import DataProfile


# ---------------------------------------------------------------------------
# Colour palette (IBM-inspired)
# ---------------------------------------------------------------------------

IBM_BLUE = "#0f62fe"
IBM_RED = "#da1e28"
IBM_GREEN = "#24a148"
IBM_PURPLE = "#8a3ffc"
IBM_CYAN = "#1192e8"
IBM_ORANGE = "#ff832b"
IBM_TEAL = "#009d9a"

PALETTE = [IBM_BLUE, IBM_RED, IBM_GREEN, IBM_PURPLE, IBM_CYAN, IBM_ORANGE, IBM_TEAL]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#1f2328"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def _apply_layout(fig, title: str = ""):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=16, color="#1f2328")))
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Target Distribution
# ---------------------------------------------------------------------------

def plot_target_distribution(profile: DataProfile):
    """Bar + pie chart of target class distribution."""
    if not profile.target_col:
        return None

    dist = profile.target_distribution
    labels = [str(k) for k in dist.keys()]
    values = list(dist.values())
    total = sum(values)
    pcts = [f"{v/total*100:.1f}%" for v in values]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=["Count per Class", "Class Proportion"],
    )

    fig.add_trace(
        go.Bar(
            x=labels, y=values,
            marker_color=PALETTE[:len(labels)],
            text=pcts, textposition="outside",
            name="Count",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=labels, values=values,
            marker=dict(colors=PALETTE[:len(labels)]),
            textinfo="percent+label",
            hole=0.4,
            name="Proportion",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Target Distribution — <b>{profile.target_col}</b>", font=dict(size=16)),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Missing Values
# ---------------------------------------------------------------------------

def plot_missing_values(profile: DataProfile):
    """Horizontal bar of missing value percentages."""
    if not profile.missing_pct:
        return None

    sorted_items = sorted(profile.missing_pct.items(), key=lambda x: x[1], reverse=True)
    cols = [i[0] for i in sorted_items]
    pcts = [i[1] for i in sorted_items]

    fig = go.Figure(go.Bar(
        x=pcts, y=cols, orientation="h",
        marker=dict(
            color=pcts,
            colorscale=[[0, IBM_GREEN], [0.5, IBM_ORANGE], [1, IBM_RED]],
            showscale=True,
            colorbar=dict(title="% Missing"),
        ),
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
    ))
    _apply_layout(fig, "Missing Values by Column")
    fig.update_layout(height=max(300, len(cols) * 28))
    return fig


# ---------------------------------------------------------------------------
# Numeric Distributions
# ---------------------------------------------------------------------------

def plot_numeric_distributions(profile: DataProfile, max_cols: int = 12):
    """Grid of histograms for numeric feature columns."""
    cols = profile.numeric_features[:max_cols]
    if not cols:
        return None

    n_cols_grid = 3
    n_rows_grid = int(np.ceil(len(cols) / n_cols_grid))

    fig = make_subplots(
        rows=n_rows_grid,
        cols=n_cols_grid,
        subplot_titles=cols,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    df = profile.df
    for i, col in enumerate(cols):
        r = i // n_cols_grid + 1
        c = i % n_cols_grid + 1
        data = df[col].dropna()
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                marker_color=PALETTE[i % len(PALETTE)],
                opacity=0.8,
                name=col,
                showlegend=False,
            ),
            row=r, col=c,
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Numeric Feature Distributions", font=dict(size=16)),
        height=n_rows_grid * 220,
    )
    return fig


# ---------------------------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(profile: DataProfile):
    """Correlation matrix for numeric features (max 20 cols)."""
    cols = profile.numeric_features[:20]
    if len(cols) < 2:
        return None

    corr = profile.df[cols].corr(numeric_only=True)
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

    z = corr.values.copy()
    z[mask_upper] = None

    fig = go.Figure(go.Heatmap(
        z=z,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, IBM_RED], [0.5, "#ffffff"], [1.0, IBM_BLUE]
        ],
        zmin=-1, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="r"),
    ))
    _apply_layout(fig, "Feature Correlation Matrix (Lower Triangle)")
    fig.update_layout(height=max(400, len(cols) * 30))
    return fig


# ---------------------------------------------------------------------------
# Categorical Counts
# ---------------------------------------------------------------------------

def plot_categorical_distributions(profile: DataProfile, max_cols: int = 8):
    """Bar charts for categorical features (top 10 values each)."""
    cols = [c for c in profile.categorical_features if c not in profile.high_cardinality_cols][:max_cols]
    if not cols:
        return None

    n_cols_grid = 2
    n_rows_grid = int(np.ceil(len(cols) / n_cols_grid))

    fig = make_subplots(
        rows=n_rows_grid,
        cols=n_cols_grid,
        subplot_titles=cols,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    df = profile.df
    for i, col in enumerate(cols):
        r = i // n_cols_grid + 1
        c = i % n_cols_grid + 1
        vc = df[col].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=vc.index.astype(str).tolist(),
                y=vc.values.tolist(),
                marker_color=PALETTE[i % len(PALETTE)],
                name=col,
                showlegend=False,
            ),
            row=r, col=c,
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Categorical Feature Distributions (Top 10 Values)", font=dict(size=16)),
        height=n_rows_grid * 250,
    )
    return fig


# ---------------------------------------------------------------------------
# Box Plots by Target
# ---------------------------------------------------------------------------

def plot_boxplots_by_target(profile: DataProfile, max_cols: int = 8):
    """Box plots of numeric features split by target class."""
    if not profile.target_col or not profile.numeric_features:
        return None

    cols = profile.numeric_features[:max_cols]
    n_cols_grid = 2
    n_rows_grid = int(np.ceil(len(cols) / n_cols_grid))

    fig = make_subplots(
        rows=n_rows_grid,
        cols=n_cols_grid,
        subplot_titles=cols,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    df = profile.df
    target = profile.target_col
    classes = [str(c) for c in profile.target_classes[:6]]

    for i, col in enumerate(cols):
        r = i // n_cols_grid + 1
        c = i % n_cols_grid + 1
        for j, cls in enumerate(classes):
            subset = df[df[target].astype(str) == cls][col].dropna()
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=f"{cls}",
                    marker_color=PALETTE[j % len(PALETTE)],
                    legendgroup=cls,
                    showlegend=(i == 0),
                ),
                row=r, col=c,
            )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Numeric Features by Target Class ({target})", font=dict(size=16)),
        height=n_rows_grid * 260,
        legend=dict(title=target),
    )
    return fig


# ---------------------------------------------------------------------------
# Outlier Detection (IQR)
# ---------------------------------------------------------------------------

def compute_outlier_summary(profile: DataProfile) -> pd.DataFrame:
    """Return a DataFrame summarising outliers per numeric feature using IQR method."""
    rows = []
    df = profile.df
    for col in profile.numeric_features:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({
            "Feature": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower Fence": round(low, 4),
            "Upper Fence": round(high, 4),
            "Outlier Count": n_out,
            "Outlier %": round(n_out / len(s) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("Outlier %", ascending=False)


# ---------------------------------------------------------------------------
# Scatter Matrix
# ---------------------------------------------------------------------------

def plot_scatter_matrix(profile: DataProfile):
    """Pair-plot for up to 5 numeric features, coloured by target."""
    if not profile.target_col:
        return None

    cols = profile.numeric_features[:5]
    if len(cols) < 2:
        return None

    df = profile.df[cols + [profile.target_col]].dropna()
    fig = px.scatter_matrix(
        df,
        dimensions=cols,
        color=profile.target_col.replace(" ", "_"),
        color_discrete_sequence=PALETTE,
        opacity=0.4,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
    _apply_layout(fig, "Scatter Matrix — Top 5 Numeric Features")
    fig.update_layout(height=600)
    return fig


# ---------------------------------------------------------------------------
# Summary Statistics Table
# ---------------------------------------------------------------------------

def compute_summary_stats(profile: DataProfile) -> pd.DataFrame:
    """Extended describe() with skewness and kurtosis."""
    if not profile.numeric_features:
        return pd.DataFrame()

    df = profile.df[profile.numeric_features]
    desc = df.describe().T
    desc["skewness"] = df.skew(numeric_only=True)
    desc["kurtosis"] = df.kurtosis(numeric_only=True)
    desc["missing"] = df.isnull().sum()
    desc["missing_%"] = (df.isnull().sum() / len(df) * 100).round(2)
    return desc.round(4)


# ---------------------------------------------------------------------------
# Class Imbalance Analysis
# ---------------------------------------------------------------------------

def compute_class_imbalance(profile: DataProfile) -> dict:
    """Return imbalance ratio and recommendation."""
    if not profile.target_col or not profile.is_binary_target:
        return {}

    dist = profile.target_distribution
    values = sorted(dist.values(), reverse=True)
    if len(values) < 2 or values[1] == 0:
        return {}

    ratio = values[0] / values[1]
    if ratio > 10:
        severity = "Severe"
        recommendation = "Use SMOTE, class_weight='balanced', or oversampling."
    elif ratio > 3:
        severity = "Moderate"
        recommendation = "Use class_weight='balanced' or threshold tuning."
    else:
        severity = "Mild / Balanced"
        recommendation = "Dataset is reasonably balanced. Standard training is fine."

    return {
        "majority_class": list(dist.keys())[list(dist.values()).index(values[0])],
        "minority_class": list(dist.keys())[list(dist.values()).index(values[1])],
        "majority_count": values[0],
        "minority_count": values[1],
        "imbalance_ratio": round(ratio, 2),
        "severity": severity,
        "recommendation": recommendation,
    }
