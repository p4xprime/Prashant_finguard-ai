"""
FinGuard AI — Excel Report Generator
Exports a multi-sheet professional report using openpyxl.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import (
        Alignment,
        Border,
        Font,
        GradientFill,
        PatternFill,
        Side,
    )
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.chart import BarChart, Reference
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from utils.data_loader import DataProfile
from utils.model_engine import ModelResult


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _header_fill(colour: str = "0F62FE"):
    return PatternFill(start_color=colour, end_color=colour, fill_type="solid")


def _cell_border():
    thin = Side(style="thin", color="D0D0D0")
    return Border(left=thin, right=thin, top=thin, bottom=thin)


def _style_header_row(ws, row_idx: int, n_cols: int, colour: str = "0F62FE"):
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=col)
        cell.fill = _header_fill(colour)
        cell.font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _cell_border()


def _style_data_row(ws, row_idx: int, n_cols: int, even: bool = False):
    bg = "F2F4FF" if even else "FFFFFF"
    fill = PatternFill(start_color=bg, end_color=bg, fill_type="solid")
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=col)
        cell.fill = fill
        cell.font = Font(name="Calibri", size=10)
        cell.alignment = Alignment(vertical="center")
        cell.border = _cell_border()


def _write_title(ws, title: str, subtitle: str = ""):
    ws.merge_cells("A1:H1")
    t_cell = ws["A1"]
    t_cell.value = title
    t_cell.font = Font(bold=True, size=16, color="0F62FE", name="Calibri")
    t_cell.alignment = Alignment(horizontal="center", vertical="center")
    t_cell.fill = PatternFill(start_color="EEF2FF", end_color="EEF2FF", fill_type="solid")
    ws.row_dimensions[1].height = 32

    if subtitle:
        ws.merge_cells("A2:H2")
        s_cell = ws["A2"]
        s_cell.value = subtitle
        s_cell.font = Font(italic=True, size=10, color="5E6278", name="Calibri")
        s_cell.alignment = Alignment(horizontal="center")
        return 3
    return 2


def _df_to_sheet(ws, df: pd.DataFrame, start_row: int = 1, colour: str = "0F62FE"):
    """Write a DataFrame to the worksheet starting at start_row."""
    # Header
    for col_idx, col_name in enumerate(df.columns, 1):
        ws.cell(row=start_row, column=col_idx, value=str(col_name))
    _style_header_row(ws, start_row, len(df.columns), colour)

    # Data
    for r_idx, row in enumerate(df.itertuples(index=False), start_row + 1):
        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx)
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = round(float(val), 4)
            elif isinstance(val, float):
                val = round(val, 4)
            cell.value = val
        _style_data_row(ws, r_idx, len(df.columns), even=(r_idx % 2 == 0))

    # Auto-width
    for col in ws.iter_cols(min_row=start_row, max_row=ws.max_row):
        max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 30)


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def _sheet_overview(wb: Workbook, profile: DataProfile, generated_at: str):
    ws = wb.create_sheet("Overview")
    next_row = _write_title(ws, "FinGuard AI — Dataset Overview", generated_at)

    rows = [
        ("Dataset Shape", f"{profile.n_rows:,} rows × {profile.n_cols} columns"),
        ("Total Features", str(len(profile.feature_cols))),
        ("Numeric Features", str(len(profile.numeric_features))),
        ("Categorical Features", str(len(profile.categorical_features))),
        ("Target Column", str(profile.target_col)),
        ("Target Classes", str(profile.target_classes)),
        ("Binary Target", str(profile.is_binary_target)),
        ("Total Missing Values", f"{profile.total_missing:,}"),
        ("Duplicate Rows", f"{profile.duplicate_rows:,}"),
        ("ID Columns Detected", ", ".join(profile.id_cols) or "None"),
        ("High Cardinality Cols", ", ".join(profile.high_cardinality_cols) or "None"),
    ]

    next_row += 1
    for label, value in rows:
        ws.cell(row=next_row, column=1, value=label).font = Font(bold=True, name="Calibri")
        ws.cell(row=next_row, column=2, value=value).font = Font(name="Calibri")
        next_row += 1

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 50


def _sheet_data_profile(wb: Workbook, profile: DataProfile):
    ws = wb.create_sheet("Data Profile")
    _write_title(ws, "Column-Level Data Profile")

    df = profile.df
    rows = []
    for col in profile.all_columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_missing = int(df[col].isnull().sum())
        pct_missing = round(n_missing / profile.n_rows * 100, 2)
        col_type = (
            "Target" if col == profile.target_col else
            "ID" if col in profile.id_cols else
            "Datetime" if col in profile.datetime_cols else
            "Numeric" if col in profile.numeric_cols else
            "Categorical"
        )
        sample_vals = str(df[col].dropna().unique()[:5].tolist())

        row = {
            "Column": col,
            "Type": col_type,
            "Dtype": dtype,
            "Unique Values": n_unique,
            "Missing Count": n_missing,
            "Missing %": pct_missing,
            "Sample Values": sample_vals[:80],
        }

        if col in profile.numeric_cols:
            s = df[col].dropna()
            row["Min"] = round(float(s.min()), 4) if len(s) > 0 else None
            row["Max"] = round(float(s.max()), 4) if len(s) > 0 else None
            row["Mean"] = round(float(s.mean()), 4) if len(s) > 0 else None
            row["Std Dev"] = round(float(s.std()), 4) if len(s) > 0 else None
        else:
            row["Min"] = row["Max"] = row["Mean"] = row["Std Dev"] = "N/A"

        rows.append(row)

    _df_to_sheet(ws, pd.DataFrame(rows), start_row=3)


def _sheet_model_results(wb: Workbook, results: dict[str, ModelResult]):
    ws = wb.create_sheet("Model Results")
    _write_title(ws, "Machine Learning Model Comparison")

    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Accuracy (%)": round(res.accuracy * 100, 2),
            "Precision (%)": round(res.precision * 100, 2),
            "Recall (%)": round(res.recall * 100, 2),
            "F1-Score (%)": round(res.f1 * 100, 2),
            "ROC-AUC (%)": round(res.roc_auc * 100, 2),
            "Avg Precision (%)": round(res.avg_precision * 100, 2),
            "CV Mean ROC-AUC (%)": round(res.cv_mean * 100, 2),
            "CV Std": round(res.cv_std, 4),
            "Train Time (s)": round(res.train_time, 3),
        })

    _df_to_sheet(ws, pd.DataFrame(rows), start_row=3, colour="24A148")


def _sheet_feature_importance(wb: Workbook, result: ModelResult, feature_names: list[str]):
    ws = wb.create_sheet("Feature Importance")
    _write_title(ws, f"Feature Importance — {result.name}")

    if result.feature_importances is None:
        ws["A3"] = "Feature importance not available for this model."
        return

    fi = result.feature_importances
    names = np.array(feature_names)
    idx = np.argsort(fi)[::-1]

    rows = [
        {"Rank": i + 1, "Feature": names[idx[i]], "Importance": round(float(fi[idx[i]]), 6)}
        for i in range(len(idx))
    ]
    _df_to_sheet(ws, pd.DataFrame(rows), start_row=3, colour="8A3FFC")


def _sheet_missing_values(wb: Workbook, profile: DataProfile):
    ws = wb.create_sheet("Missing Values")
    _write_title(ws, "Missing Value Analysis")

    if not profile.missing_pct:
        ws["A3"] = "✓ No missing values detected in the dataset."
        ws["A3"].font = Font(color="24A148", bold=True, name="Calibri")
        return

    rows = [
        {"Column": col, "Missing Count": cnt, "Missing %": pct}
        for (col, cnt), pct in zip(
            profile.missing_counts.items(), profile.missing_pct.values()
        )
    ]
    _df_to_sheet(ws, pd.DataFrame(rows), start_row=3, colour="DA1E28")


def _sheet_class_distribution(wb: Workbook, profile: DataProfile):
    ws = wb.create_sheet("Class Distribution")
    _write_title(ws, f"Target Distribution — {profile.target_col}")

    dist = profile.target_distribution
    total = sum(dist.values())
    rows = [
        {
            "Class": str(k),
            "Count": v,
            "Percentage (%)": round(v / total * 100, 2),
        }
        for k, v in dist.items()
    ]
    _df_to_sheet(ws, pd.DataFrame(rows), start_row=3, colour="FF832B")


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def generate_excel_report(
    profile: DataProfile,
    results: dict[str, ModelResult],
    best_model: ModelResult,
    feature_names: list[str],
) -> bytes:
    """Generate a multi-sheet Excel report and return as bytes."""
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl is required for report generation. Install it with: pip install openpyxl")

    wb = Workbook()
    wb.remove(wb.active)  # remove default sheet

    generated_at = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | FinGuard AI"

    _sheet_overview(wb, profile, generated_at)
    _sheet_data_profile(wb, profile)
    _sheet_class_distribution(wb, profile)
    _sheet_missing_values(wb, profile)
    _sheet_model_results(wb, results)
    _sheet_feature_importance(wb, best_model, feature_names)

    # Summary sheet (last)
    ws_summary = wb.create_sheet("Executive Summary", 0)
    _write_title(ws_summary, "FinGuard AI — Executive Summary Report", generated_at)
    ws_summary.merge_cells("A3:H3")
    ws_summary["A3"] = "Bank Credit & Loan Default Risk Analytics Engine"
    ws_summary["A3"].font = Font(bold=True, size=14, color="1F2328", name="Calibri")
    ws_summary["A3"].alignment = Alignment(horizontal="center")

    summary_items = [
        ("Project", "FinGuard AI"),
        ("Student", "Prashant"),
        ("Program", "IBM SkillsBuild Data Analytics with AI — BharatCares & AICTE"),
        ("Domain", "Finance & Banking — Credit Risk"),
        ("Dataset Rows", f"{profile.n_rows:,}"),
        ("Dataset Columns", str(profile.n_cols)),
        ("Target Variable", str(profile.target_col)),
        ("Best Model", best_model.name),
        ("Best ROC-AUC", f"{best_model.roc_auc*100:.2f}%"),
        ("Best F1-Score", f"{best_model.f1*100:.2f}%"),
        ("Best Accuracy", f"{best_model.accuracy*100:.2f}%"),
        ("Total Models Evaluated", str(len(results))),
    ]

    for i, (k, v) in enumerate(summary_items, start=5):
        ws_summary.cell(row=i, column=2, value=k).font = Font(bold=True, name="Calibri", size=11)
        ws_summary.cell(row=i, column=3, value=v).font = Font(name="Calibri", size=11)

    ws_summary.column_dimensions["B"].width = 32
    ws_summary.column_dimensions["C"].width = 50

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()
