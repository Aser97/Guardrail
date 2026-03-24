"""Compute precision, recall, F1 and latency for guardrails (single or stacked).

Works with any guardrail that implements the guardrail protocol (e.g. LLM Judge,
ClassifierGuardrail). For stacked guardrails, predictions are combined: the
stack is considered to predict "high_risk" if any guardrail in the stack returns
a high_risk result; then metrics are computed on that combined prediction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from sklearn.metrics import precision_score, recall_score, f1_score

from .base import GuardrailProtocol, GuardrailStatus, EvaluationType
from .submission_loader import describe_guardrail

LOGGER = logging.getLogger(__name__)


# Single guardrail or a stack (sequence) of guardrails
GuardrailOrStack = Union[GuardrailProtocol, Sequence[GuardrailProtocol]]


def _normalize_guardrail_or_stack(
    guardrail: Optional[GuardrailOrStack],
) -> List[GuardrailProtocol]:
    """Normalize a single guardrail or sequence to a list (empty if None)."""
    if guardrail is None:
        return []
    if isinstance(guardrail, (list, tuple)):
        return list(guardrail)
    return [guardrail]


def _label_to_bool(label: Any) -> bool:
    """Convert various label formats to bool (True = high_risk)."""
    if isinstance(label, bool):
        return label
    if isinstance(label, int):
        return label != 0
    if isinstance(label, str):
        v = label.strip().lower()
        if v in ("yes", "true", "1", "high_risk"):
            return True
        if v in ("no", "false", "0", "low_risk"):
            return False
    return bool(label)


@dataclass
class GuardrailMetricsResult:
    """Result of guardrail metrics computation."""

    precision: float
    recall: float
    f1: float
    support_high_risk: int  # number of true high_risk samples
    support_low_risk: int     # number of true low_risk samples
    total_samples: int
    # Latency (ms): for the whole stack per sample
    latency_ms_mean: Optional[float] = None
    latency_ms_total: Optional[float] = None
    latency_ms_per_sample: Optional[List[float]] = None
    # Optional: per-guardrail names (for stacked)
    guardrail_names: List[str] = field(default_factory=list)


def _sanitize_csv_column(name: str) -> str:
    """Make a string low_risk for use as a CSV column header (no commas, newlines)."""
    return (name or "").replace(",", "_").replace("\n", " ").replace("\r", " ").strip() or "guardrail"


def get_predictions(
    guardrail: GuardrailOrStack,
    evaluation_data: List[Dict[str, Any]],
    evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    context: Optional[Dict[str, Any]] = None,
    *,
    content_key: str = "content",
    label_key: str = "label",
    include_latency: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a guardrail (or stack) on labeled data and return per-sample predictions.

    For stacked guardrails: runs all guardrails on each sample (no short-circuit),
    then combines so that the stack predicts "high_risk" if any guardrail returns
    a high_risk result. Use compute_metrics_from_predictions() to get precision,
    recall, F1 and latency from the returned list.

    Args:
        guardrail: A single guardrail or a list/tuple of guardrails (stack).
        evaluation_data: List of dicts. Each must have content (or content_key)
            and label (or label_key). Optional: "evaluation_type" per row.
        evaluation_type: Default evaluation type (USER_INPUT).
        context: Optional context passed to evaluate().
        content_key: Key in each row for the text to evaluate.
        label_key: Key in each row for the ground-truth label (True/1 = high_risk).
        include_latency: Whether to include latency_ms per sample (inference time only).

    Returns:
        List of dicts per sample with keys: content, label, label_high_risk,
        one key per guardrail (sanitized name) with bool, combined_pred (bool),
        and optionally latency_ms.
    """
    stack = _normalize_guardrail_or_stack(guardrail)
    if not stack:
        LOGGER.warning("get_predictions called with no guardrails; returning empty list")
        return []

    guardrail_names_sanitized = [
        _sanitize_csv_column(
            getattr(g, "config", None) and getattr(g.config, "name", "") or type(g).__name__
        )
        for g in stack
    ]
    seen: Dict[str, int] = {}
    unique_names: List[str] = []
    for n in guardrail_names_sanitized:
        count = seen.get(n, 0)
        seen[n] = count + 1
        unique_names.append(f"{n}_{count}" if count else n)

    total_samples = len(evaluation_data)
    LOGGER.info(
        "Running predictions | samples=%d guardrails=%d names=%s",
        total_samples,
        len(stack),
        unique_names,
    )
    LOGGER.info(
        "Prediction guardrails resolved | %s",
        [describe_guardrail(g) for g in stack],
    )

    predictions_list: List[Dict[str, Any]] = []
    error_count = 0
    progress_interval = max(1, total_samples // 10)

    for idx, row in enumerate(evaluation_data):
        content = row.get(content_key) or row.get("text") or row.get("prompt")
        if content is None:
            LOGGER.error(
                "Sample %d missing content | tried keys: %s, 'text', 'prompt'",
                idx,
                content_key,
            )
            content = ""
        label = row.get(label_key) or row.get("is_high_risk")
        label_high_risk = _label_to_bool(label)

        et = row.get("evaluation_type")
        if isinstance(et, EvaluationType):
            ev_type = et
        elif isinstance(et, str):
            LOGGER.warning(
                "Sample %d has string evaluation_type=%r; defaulting to %s",
                idx,
                et,
                EvaluationType.USER_INPUT.value,
            )
            ev_type = EvaluationType.USER_INPUT
        else:
            ev_type = evaluation_type

        any_high_risk = False
        pred_row: Dict[str, Any] = {
            "content": str(content).strip(),
            "label": label,
            "label_high_risk": label_high_risk,
        }
        # Latency = inference time only: time spent in guardrail evaluate() calls
        t0 = time.perf_counter()
        for gr, col_name in zip(stack, unique_names):
            try:
                result = gr.evaluate(
                    content=str(content).strip(),
                    context=context,
                    evaluation_type=ev_type,
                )
            except Exception:
                LOGGER.exception(
                    "Guardrail evaluate raised exception | sample=%d guardrail=%s evaluation_type=%s content_chars=%d",
                    idx,
                    col_name,
                    ev_type.value,
                    len(str(content).strip()),
                )
                raise
            if isinstance(result, dict):
                status = result.get("status")
                if isinstance(status, GuardrailStatus):
                    is_high_risk = status != GuardrailStatus.PASS
                    if status == GuardrailStatus.ERROR:
                        error_count += 1
                        LOGGER.warning(
                            "Sample %d: guardrail '%s' returned ERROR | reason=%s",
                            idx,
                            col_name,
                            result.get("reasoning", "unknown"),
                        )
                else:
                    is_high_risk = str(status).lower() == "fail"
            else:
                if not hasattr(result, "is_high_risk") or not hasattr(result, "status"):
                    LOGGER.error(
                        "Guardrail returned unexpected result type | sample=%d guardrail=%s result_type=%s result=%r",
                        idx,
                        col_name,
                        type(result).__name__,
                        result,
                    )
                    raise TypeError(
                        f"Guardrail '{col_name}' returned unsupported result type: {type(result).__name__}"
                    )
                is_high_risk = result.is_high_risk
                if result.status == GuardrailStatus.ERROR:
                    error_count += 1
                    LOGGER.warning(
                        "Sample %d: guardrail '%s' returned ERROR | reason=%s",
                        idx,
                        col_name,
                        result.reasoning or "unknown",
                    )
            any_high_risk = any_high_risk or is_high_risk
            pred_row[col_name] = is_high_risk
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pred_row["combined_pred"] = any_high_risk
        if include_latency:
            pred_row["latency_ms"] = round(elapsed_ms, 4)
        predictions_list.append(pred_row)

        if (idx + 1) % progress_interval == 0:
            LOGGER.info("Progress: %d/%d samples processed", idx + 1, total_samples)

    high_risk_count = sum(1 for p in predictions_list if p.get("combined_pred"))
    total_latency = sum(p.get("latency_ms", 0) for p in predictions_list)
    mean_latency = total_latency / len(predictions_list) if predictions_list else 0

    LOGGER.info(
        "Predictions complete | samples=%d predicted_high_risk=%d errors=%d mean_latency_ms=%.2f",
        total_samples,
        high_risk_count,
        error_count,
        mean_latency,
    )

    return predictions_list


def _pred_to_bool(v: Any) -> bool:
    """Convert a prediction value from CSV or dict to bool (True = high_risk)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "high_risk")
    return bool(v)


def compute_metrics_from_predictions(
    predictions: List[Dict[str, Any]],
    *,
    combined_pred_key: str = "combined_pred",
    label_key: str = "label_high_risk",
    fallback_label_key: str = "label",
    latency_key: str = "latency_ms",
    guardrail_names: Optional[List[str]] = None,
) -> GuardrailMetricsResult:
    """
    Compute precision, recall, F1 and optional latency from a list of prediction rows.

    Use this when you have already run guardrails and have per-sample predictions
    (e.g. from get_predictions or a previous run). Each row must have the
    ground-truth label (high_risk or not) and the combined prediction.

    Args:
        predictions: List of dicts. Each must have combined_pred (or combined_pred_key)
            and label (label_key or fallback_label_key). Optional: latency_key for
            per-sample latency in ms.
        combined_pred_key: Key for the guardrail(s) combined prediction (bool or 0/1).
        label_key: Key for ground-truth high_risk flag (bool or 0/1). If missing,
            fallback_label_key is used and converted via _label_to_bool.
        fallback_label_key: Key for raw label when label_key is absent.
        latency_key: Key for per-sample latency in ms (optional).
        guardrail_names: Optional list of guardrail names for the result.

    Returns:
        GuardrailMetricsResult with precision, recall, F1, support, and latency
        stats if latency_key is present in the rows.
    """
    LOGGER.info("Computing metrics from %d predictions", len(predictions))

    if not predictions:
        LOGGER.warning("No predictions provided; returning zero metrics")
        return GuardrailMetricsResult(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            support_high_risk=0,
            support_low_risk=0,
            total_samples=0,
            guardrail_names=guardrail_names or [],
        )

    y_true: List[bool] = []
    y_pred: List[bool] = []
    latencies_ms: List[float] = []

    for row in predictions:
        label_val = row.get(label_key)
        if label_val is None:
            label_val = row.get(fallback_label_key)
        # Always use _label_to_bool for ground-truth so "low_risk"/"high_risk"/"yes"/"no" etc. are correct
        label_high_risk = _label_to_bool(label_val)
        pred_val = row.get(combined_pred_key)
        combined_pred = _pred_to_bool(pred_val) if pred_val is not None else False

        y_true.append(label_high_risk)
        y_pred.append(combined_pred)

        if latency_key in row and row[latency_key] is not None:
            try:
                latencies_ms.append(float(row[latency_key]))
            except (TypeError, ValueError):
                pass

    support_high_risk = sum(y_true)
    support_low_risk = len(y_true) - support_high_risk

    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    latency_ms_mean = (
        float(sum(latencies_ms)) / len(latencies_ms) if latencies_ms else None
    )
    latency_ms_total = sum(latencies_ms) if latencies_ms else None

    LOGGER.info(
        "Metrics computed | precision=%.4f recall=%.4f f1=%.4f | "
        "support: %d high_risk, %d low_risk | latency_mean_ms=%s",
        precision,
        recall,
        f1,
        support_high_risk,
        support_low_risk,
        f"{latency_ms_mean:.2f}" if latency_ms_mean is not None else "N/A",
    )

    return GuardrailMetricsResult(
        precision=precision,
        recall=recall,
        f1=f1,
        support_high_risk=support_high_risk,
        support_low_risk=support_low_risk,
        total_samples=len(y_true),
        latency_ms_mean=latency_ms_mean,
        latency_ms_total=latency_ms_total,
        latency_ms_per_sample=latencies_ms if latencies_ms else None,
        guardrail_names=guardrail_names or [],
    )
