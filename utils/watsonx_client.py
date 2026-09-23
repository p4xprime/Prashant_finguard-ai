"""
FinGuard AI — IBM watsonx.ai Integration
Provides AI-powered explanations, risk narratives, and insights
using IBM Granite foundation models.
"""

from __future__ import annotations

import os
from typing import Optional


# ---------------------------------------------------------------------------
# Lazy import — only required when user has watsonx configured
# ---------------------------------------------------------------------------

def _get_watsonx_model(api_key: str, project_id: str, url: str):
    """Initialise a watsonx.ai ModelInference instance."""
    from ibm_watsonx_ai import APIClient, Credentials  # type: ignore[import-untyped]
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore[import-untyped]
    from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  # type: ignore[import-untyped]

    credentials = Credentials(url=url, api_key=api_key)
    client = APIClient(credentials)

    model = ModelInference(
        model_id="ibm/granite-13b-instruct-v2",
        api_client=client,
        project_id=project_id,
        params={
            "max_new_tokens": 600,
            "min_new_tokens": 50,
            "decoding_method": "greedy",
            "repetition_penalty": 1.1,
            "stop_sequences": ["<|endoftext|>"],
        },
    )
    return model


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_risk_explanation_prompt(
    applicant_data: dict,
    predicted_class: int,
    probability: float,
    risk_label: str,
    top_features: list[tuple[str, float]],
    dataset_name: str = "credit dataset",
) -> str:
    feat_lines = "\n".join(
        f"  - {name}: {val:.4f}" for name, val in top_features[:8]
    )
    prediction_text = (
        "DEFAULT (high risk)" if predicted_class == 1 else "NON-DEFAULT (low risk)"
    )
    data_lines = "\n".join(f"  - {k}: {v}" for k, v in list(applicant_data.items())[:15])

    return f"""You are FinGuard AI, an expert credit risk analyst at a leading financial institution.
Analyze the following loan applicant's risk profile and provide a clear, professional explanation.

APPLICANT DATA ({dataset_name}):
{data_lines}

MODEL PREDICTION: {prediction_text}
DEFAULT PROBABILITY: {probability:.1%}
RISK TIER: {risk_label}

TOP CONTRIBUTING FEATURES:
{feat_lines}

Provide a structured risk assessment with:
1. Executive Summary (2–3 sentences)
2. Key Risk Drivers (bullet points, referencing the actual features above)
3. Mitigating Factors (if any)
4. Recommended Decision (Approve / Conditional Approve / Reject)
5. Suggested Actions for Risk Mitigation

Be concise, factual, and avoid speculation. Reference only the provided features.
Response:"""


def _build_eda_insights_prompt(
    dataset_summary: dict,
    target_col: str,
    imbalance_info: dict,
    top_correlations: list[tuple[str, str, float]],
) -> str:
    corr_lines = "\n".join(
        f"  - {a} ↔ {b}: {r:.3f}" for a, b, r in top_correlations[:5]
    )
    imbalance_text = (
        f"Imbalance ratio: {imbalance_info.get('imbalance_ratio', 'N/A')} "
        f"({imbalance_info.get('severity', 'unknown')})"
        if imbalance_info else "Not computed"
    )

    return f"""You are FinGuard AI, a senior data scientist specialising in credit risk analytics.
Based on the following dataset statistics, provide actionable EDA insights.

DATASET OVERVIEW:
  - Rows: {dataset_summary.get('rows', 'N/A')}
  - Columns: {dataset_summary.get('cols', 'N/A')}
  - Missing values: {dataset_summary.get('missing_pct', 'N/A')}%
  - Duplicate rows: {dataset_summary.get('duplicates', 'N/A')}
  - Target column: {target_col}
  - Class imbalance: {imbalance_text}

TOP FEATURE CORRELATIONS WITH TARGET:
{corr_lines if corr_lines else "  (Not available — non-numeric target or insufficient numeric features)"}

Provide:
1. Data Quality Assessment (3–4 bullet points)
2. Key Patterns and Insights from the statistics
3. Class Imbalance Analysis and recommendation
4. Feature Engineering suggestions (based only on available columns)
5. Modelling Strategy recommendation

Be specific, concise, and actionable.
Response:"""


def _build_model_insights_prompt(
    best_model_name: str,
    best_roc_auc: float,
    best_f1: float,
    all_model_scores: list[dict],
    dataset_rows: int,
    is_imbalanced: bool,
) -> str:
    scores_text = "\n".join(
        f"  - {m['name']}: ROC-AUC={m['roc_auc']:.3f}, F1={m['f1']:.3f}, Acc={m['accuracy']:.3f}"
        for m in all_model_scores
    )
    return f"""You are FinGuard AI, a machine learning expert in credit risk modelling.
Summarise and interpret the following model evaluation results.

BEST MODEL: {best_model_name}
BEST ROC-AUC: {best_roc_auc:.3f}
BEST F1-SCORE: {best_f1:.3f}
DATASET SIZE: {dataset_rows} rows
CLASS IMBALANCE: {'Yes' if is_imbalanced else 'No'}

ALL MODELS:
{scores_text}

Provide:
1. Best Model Analysis — why it performed best (2–3 sentences)
2. Performance Interpretation — what ROC-AUC and F1 mean for credit risk
3. Model Weaknesses and potential failure modes
4. Deployment Recommendation — which model to use in production and why
5. Next Steps for model improvement

Focus on practical credit risk deployment considerations.
Response:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class WatsonxClient:
    """Wrapper around watsonx.ai for FinGuard AI insights."""

    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str = "https://us-south.ml.cloud.ibm.com",
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        self._model = None
        self._connected = False
        self._error: Optional[str] = None

    def connect(self) -> bool:
        """Attempt to connect and return True on success."""
        try:
            self._model = _get_watsonx_model(self.api_key, self.project_id, self.url)
            # Warm-up ping
            self._model.generate_text("Hello")
            self._connected = True
            return True
        except Exception as e:
            self._error = str(e)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt. Raises if not connected."""
        if not self._connected or self._model is None:
            raise RuntimeError("watsonx.ai client is not connected. Call connect() first.")
        try:
            return self._model.generate_text(prompt=prompt)
        except Exception as e:
            raise RuntimeError(f"watsonx.ai generation failed: {e}") from e

    def explain_risk(
        self,
        applicant_data: dict,
        predicted_class: int,
        probability: float,
        risk_label: str,
        top_features: list[tuple[str, float]],
        dataset_name: str = "credit dataset",
    ) -> str:
        prompt = _build_risk_explanation_prompt(
            applicant_data, predicted_class, probability,
            risk_label, top_features, dataset_name
        )
        return self.generate(prompt)

    def generate_eda_insights(
        self,
        dataset_summary: dict,
        target_col: str,
        imbalance_info: dict,
        top_correlations: list[tuple[str, str, float]],
    ) -> str:
        prompt = _build_eda_insights_prompt(
            dataset_summary, target_col, imbalance_info, top_correlations
        )
        return self.generate(prompt)

    def generate_model_insights(
        self,
        best_model_name: str,
        best_roc_auc: float,
        best_f1: float,
        all_model_scores: list[dict],
        dataset_rows: int,
        is_imbalanced: bool,
    ) -> str:
        prompt = _build_model_insights_prompt(
            best_model_name, best_roc_auc, best_f1,
            all_model_scores, dataset_rows, is_imbalanced
        )
        return self.generate(prompt)
