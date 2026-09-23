"""
FinGuard AI — Data Loader & Auto-Detection Engine
Inspects the actual uploaded dataset, detects columns, types, target variable,
and produces a canonical data profile without fabricating any structure.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data Profile
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """Holds everything discovered about the uploaded dataset."""

    df: pd.DataFrame

    # Shape
    n_rows: int = 0
    n_cols: int = 0

    # Column classifications
    all_columns: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    binary_cols: list[str] = field(default_factory=list)
    datetime_cols: list[str] = field(default_factory=list)
    id_cols: list[str] = field(default_factory=list)
    high_cardinality_cols: list[str] = field(default_factory=list)

    # Target
    target_col: Optional[str] = None
    target_classes: list = field(default_factory=list)
    target_distribution: dict = field(default_factory=dict)
    is_binary_target: bool = False

    # Missing values
    missing_counts: dict = field(default_factory=dict)
    missing_pct: dict = field(default_factory=dict)
    total_missing: int = 0

    # Duplicates
    duplicate_rows: int = 0

    # Feature columns (everything except target + id)
    feature_cols: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)

    # Derived columns (calculated from existing data)
    derived_cols: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel from a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}")


# ---------------------------------------------------------------------------
# Target Detection
# ---------------------------------------------------------------------------

_TARGET_KEYWORDS = [
    "default", "defaulted", "loan_default", "credit_default",
    "bad_loan", "charged_off", "status", "target", "label",
    "outcome", "delinquent", "repayment", "paid", "failed",
    "fraud", "risk", "approved",
]


def _detect_target(df: pd.DataFrame) -> Optional[str]:
    """Heuristically detect the target column."""
    cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}

    # Exact keyword match (highest priority)
    for kw in _TARGET_KEYWORDS:
        if kw in cols_lower:
            return cols_lower[kw]

    # Substring match
    for col_key, col_orig in cols_lower.items():
        for kw in _TARGET_KEYWORDS:
            if kw in col_key:
                return col_orig

    # Binary numeric column with 0/1 values (fallback)
    for col in df.select_dtypes(include=[np.number]).columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return col

    return None


# ---------------------------------------------------------------------------
# ID Column Detection
# ---------------------------------------------------------------------------

_ID_KEYWORDS = ["id", "uid", "uuid", "customer_id", "loan_id", "account_id", "applicant_id", "index"]


def _detect_id_cols(df: pd.DataFrame) -> list[str]:
    """Detect likely identifier columns."""
    id_cols = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        # Name-based (keyword match only)
        if any(kw == col_lower or col_lower.endswith(f"_{kw}") or col_lower.startswith(f"{kw}_")
               for kw in _ID_KEYWORDS):
            id_cols.append(col)
    return list(dict.fromkeys(id_cols))  # preserve order, deduplicate


# ---------------------------------------------------------------------------
# Derived Feature Detection
# ---------------------------------------------------------------------------

def _detect_derived_features(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, str]:
    """
    Check if common derived metrics can be mathematically calculated
    from existing columns and record the formula description.
    """
    derived = {}
    cols_lower = {c.lower().replace(" ", "_"): c for c in numeric_cols}

    # Debt-to-Income Ratio
    debt_candidates = [c for k, c in cols_lower.items() if "debt" in k or "obligation" in k]
    income_candidates = [c for k, c in cols_lower.items() if "income" in k or "salary" in k or "earning" in k]
    if debt_candidates and income_candidates:
        derived["debt_to_income_ratio"] = (
            f"Calculated as {debt_candidates[0]} / {income_candidates[0]}"
        )

    # Loan-to-Value Ratio
    loan_candidates = [c for k, c in cols_lower.items() if "loan" in k and ("amount" in k or "value" in k)]
    property_candidates = [c for k, c in cols_lower.items() if "property" in k or "collateral" in k or "asset" in k]
    if loan_candidates and property_candidates:
        derived["loan_to_value_ratio"] = (
            f"Calculated as {loan_candidates[0]} / {property_candidates[0]}"
        )

    # Credit Utilisation
    balance_candidates = [c for k, c in cols_lower.items() if "balance" in k or "used" in k or "outstanding" in k]
    limit_candidates = [c for k, c in cols_lower.items() if "limit" in k or "credit_limit" in k]
    if balance_candidates and limit_candidates:
        derived["credit_utilisation"] = (
            f"Calculated as {balance_candidates[0]} / {limit_candidates[0]}"
        )

    return derived


# ---------------------------------------------------------------------------
# Main Profile Function
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> DataProfile:
    """Full auto-detection pipeline."""
    profile = DataProfile(df=df)

    profile.n_rows, profile.n_cols = df.shape
    profile.all_columns = list(df.columns)

    # Missing values
    missing = df.isnull().sum()
    profile.missing_counts = missing[missing > 0].to_dict()
    profile.missing_pct = {
        k: round(v / profile.n_rows * 100, 2)
        for k, v in profile.missing_counts.items()
    }
    profile.total_missing = int(missing.sum())

    # Duplicates
    profile.duplicate_rows = int(df.duplicated().sum())

    # Column type classification
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_non_null = df[col].notna().sum()

        if pd.api.types.is_datetime64_any_dtype(dtype):
            profile.datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            profile.numeric_cols.append(col)
            if n_unique == 2:
                profile.binary_cols.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            profile.categorical_cols.append(col)
            if n_unique == 2:
                profile.binary_cols.append(col)
            if n_unique > 50:
                profile.high_cardinality_cols.append(col)

    # ID columns
    profile.id_cols = _detect_id_cols(df)

    # Target detection
    profile.target_col = _detect_target(df)
    if profile.target_col:
        t_series = df[profile.target_col].dropna()
        profile.target_classes = sorted(t_series.unique().tolist())
        vc = t_series.value_counts()
        profile.target_distribution = vc.to_dict()
        profile.is_binary_target = len(profile.target_classes) == 2

    # Feature columns (exclude target and IDs)
    exclude = set(profile.id_cols)
    if profile.target_col:
        exclude.add(profile.target_col)
    profile.feature_cols = [c for c in df.columns if c not in exclude]
    profile.numeric_features = [c for c in profile.feature_cols if c in profile.numeric_cols]
    profile.categorical_features = [c for c in profile.feature_cols if c in profile.categorical_cols]

    # Derived features
    profile.derived_cols = _detect_derived_features(df, profile.numeric_features)

    return profile


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def preprocess_for_model(df: pd.DataFrame, profile: DataProfile):
    """
    Returns X (features), y (target), and the feature names list.
    Performs minimal, safe preprocessing:
      - Drops ID and datetime columns
      - Label-encodes low-cardinality categoricals
      - One-hot encodes the rest (up to 20 unique values)
      - Fills numeric NaNs with median
      - Fills categorical NaNs with mode
    """
    from sklearn.preprocessing import LabelEncoder

    if not profile.target_col:
        raise ValueError("No target column detected. Cannot train model.")

    work = df.copy()

    # Drop IDs and datetime
    drop_cols = list(set(profile.id_cols) | set(profile.datetime_cols))
    work.drop(columns=[c for c in drop_cols if c in work.columns], inplace=True)

    # Separate target
    y_raw = work.pop(profile.target_col)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    target_classes = le.classes_.tolist()

    # Fill numeric NaNs (pandas 2.x CoW-safe)
    for col in work.select_dtypes(include=[np.number]).columns:
        work[col] = work[col].fillna(work[col].median())

    # Handle categoricals
    cat_cols = work.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        fill_val = work[col].mode()[0] if not work[col].mode().empty else "Unknown"
        work[col] = work[col].fillna(fill_val)

    # One-hot encode low-cardinality; drop high-cardinality
    low_card = [c for c in cat_cols if work[c].nunique() <= 20]
    high_card = [c for c in cat_cols if work[c].nunique() > 20]
    work.drop(columns=high_card, inplace=True)

    if low_card:
        work = pd.get_dummies(work, columns=low_card, drop_first=True)

    # Convert all bool columns to int
    bool_cols = work.select_dtypes(include=["bool"]).columns
    work[bool_cols] = work[bool_cols].astype(int)

    return work.values, y, list(work.columns), target_classes
