"""Unit tests for guardrail predictions and metrics (precision, recall, F1)."""

import pytest
from unittest.mock import Mock

from src.guardrails.base import GuardrailResult, GuardrailStatus, GuardrailConfig
from src.guardrails.metrics import (
    get_predictions,
    compute_metrics_from_predictions,
)


# =============================================================================
# compute_metrics_from_predictions — metrics correctness
# =============================================================================


def test_metrics_empty_predictions():
    """Empty predictions list returns zero metrics."""
    result = compute_metrics_from_predictions([])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.support_high_risk == 0
    assert result.support_low_risk == 0
    assert result.total_samples == 0


def test_metrics_perfect_predictions():
    """When all predictions match labels, precision=1, recall=1, F1=1."""
    predictions = [
        {"label_high_risk": True, "combined_pred": True},
        {"label_high_risk": False, "combined_pred": False},
        {"label_high_risk": True, "combined_pred": True},
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.support_high_risk == 2
    assert result.support_low_risk == 1
    assert result.total_samples == 3


def test_metrics_known_tp_fp_fn():
    """Metrics match hand-computed TP, FP, FN: 2 TP, 1 FP, 2 FN."""
    # True labels:  [T, T, T, F, F]  -> 3 high_risk, 2 low_risk
    # Predicted:   [T, T, F, T, F]  -> TP=2, FP=1, FN=1
    # Precision = 2/(2+1) = 2/3, Recall = 2/(2+1) = 2/3, F1 = 2*2/3*2/3 / (4/3) = 2/3
    predictions = [
        {"label_high_risk": True, "combined_pred": True},   # TP
        {"label_high_risk": True, "combined_pred": True},   # TP
        {"label_high_risk": True, "combined_pred": False},  # FN
        {"label_high_risk": False, "combined_pred": True},  # FP
        {"label_high_risk": False, "combined_pred": False}, # TN
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.support_high_risk == 3
    assert result.support_low_risk == 2
    assert result.total_samples == 5
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)


def test_metrics_zero_recall():
    """When no high_risk is predicted, recall=0; precision depends on FP."""
    predictions = [
        {"label_high_risk": True, "combined_pred": False},
        {"label_high_risk": True, "combined_pred": False},
        {"label_high_risk": False, "combined_pred": False},
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.recall == 0.0
    assert result.precision == 0.0  # TP=0, FP=0 -> 0
    assert result.f1 == 0.0


def test_metrics_zero_precision():
    """When we predict high_risk for low_risk only, precision=0."""
    predictions = [
        {"label_high_risk": False, "combined_pred": True},
        {"label_high_risk": False, "combined_pred": True},
        {"label_high_risk": True, "combined_pred": False},
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_metrics_label_formats_high_risk_low_risk_strings():
    """Ground-truth labels 'low_risk' and 'high_risk' (and variants) are parsed correctly."""
    predictions = [
        {"label_high_risk": "high_risk", "combined_pred": True},
        {"label_high_risk": "low_risk", "combined_pred": False},
        {"label_high_risk": "high_risk", "combined_pred": True},
        {"label_high_risk": "low_risk", "combined_pred": False},
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.support_high_risk == 2
    assert result.support_low_risk == 2
    assert result.total_samples == 4
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_metrics_label_fallback_key():
    """When label_high_risk is missing, fallback to 'label' key."""
    predictions = [
        {"label": "yes", "combined_pred": True},
        {"label": "no", "combined_pred": False},
    ]
    result = compute_metrics_from_predictions(
        predictions,
        label_key="label_high_risk",
        fallback_label_key="label",
    )
    assert result.support_high_risk == 1
    assert result.support_low_risk == 1
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_metrics_label_int_and_bool():
    """Labels as int (0/1) and bool are handled correctly."""
    predictions = [
        {"label_high_risk": 1, "combined_pred": True},
        {"label_high_risk": 0, "combined_pred": False},
        {"label_high_risk": True, "combined_pred": True},
        {"label_high_risk": False, "combined_pred": False},
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.support_high_risk == 2
    assert result.support_low_risk == 2
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_metrics_custom_combined_pred_key():
    """Custom combined_pred key is respected."""
    predictions = [
        {"label_high_risk": True, "my_pred": True},
        {"label_high_risk": False, "my_pred": False},
    ]
    result = compute_metrics_from_predictions(
        predictions,
        combined_pred_key="my_pred",
    )
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_metrics_latency_aggregation():
    """Latency mean and total are computed when latency_ms present."""
    predictions = [
        {"label_high_risk": True, "combined_pred": True, "latency_ms": 10.0},
        {"label_high_risk": False, "combined_pred": False, "latency_ms": 20.0},
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.latency_ms_mean == 15.0
    assert result.latency_ms_total == 30.0
    assert result.latency_ms_per_sample == [10.0, 20.0]


def test_metrics_latency_optional():
    """When latency_ms is missing, latency fields are None."""
    predictions = [
        {"label_high_risk": True, "combined_pred": True},
    ]
    result = compute_metrics_from_predictions(predictions)
    assert result.latency_ms_mean is None
    assert result.latency_ms_total is None
    assert result.latency_ms_per_sample is None


def test_metrics_guardrail_names_passed_through():
    """Optional guardrail_names appear in result."""
    predictions = [{"label_high_risk": True, "combined_pred": True}]
    result = compute_metrics_from_predictions(
        predictions,
        guardrail_names=["gr1", "gr2"],
    )
    assert result.guardrail_names == ["gr1", "gr2"]


# =============================================================================
# get_predictions — prediction structure and combined_pred
# =============================================================================


def _make_mock_guardrail(name: str, return_pass: bool):
    """Guardrail that always returns PASS or FAIL."""
    mock = Mock()
    mock.config = GuardrailConfig(name=name, description="test", threshold=0.5)
    status = GuardrailStatus.PASS if return_pass else GuardrailStatus.FAIL
    mock.evaluate = Mock(return_value=GuardrailResult(status=status))
    return mock


def test_get_predictions_single_guardrail_pass():
    """Single guardrail returning PASS yields combined_pred False."""
    gr = _make_mock_guardrail("test_gr", return_pass=True)
    data = [{"content": "hello", "label": 0}]
    out = get_predictions(gr, data, include_latency=False)
    assert len(out) == 1
    assert out[0]["label_high_risk"] is False
    assert out[0]["combined_pred"] is False
    assert "latency_ms" not in out[0]
    gr.evaluate.assert_called_once()


def test_get_predictions_single_guardrail_fail():
    """Single guardrail returning FAIL yields combined_pred True."""
    gr = _make_mock_guardrail("test_gr", return_pass=False)
    data = [{"content": "bad", "label": 1}]
    out = get_predictions(gr, data, include_latency=False)
    assert len(out) == 1
    assert out[0]["combined_pred"] is True


def test_get_predictions_stack_any_high_risk():
    """Stack: if any guardrail says high_risk, combined_pred is True."""
    gr1 = _make_mock_guardrail("gr1", return_pass=True)
    gr2 = _make_mock_guardrail("gr2", return_pass=False)
    data = [{"content": "x", "label": 0}]
    out = get_predictions([gr1, gr2], data, include_latency=False)
    assert len(out) == 1
    assert out[0]["combined_pred"] is True
    assert out[0]["gr1"] is False
    assert out[0]["gr2"] is True


def test_get_predictions_stack_all_pass():
    """Stack: if all pass, combined_pred is False."""
    gr1 = _make_mock_guardrail("gr1", return_pass=True)
    gr2 = _make_mock_guardrail("gr2", return_pass=True)
    data = [{"content": "x", "label": 0}]
    out = get_predictions([gr1, gr2], data, include_latency=False)
    assert len(out) == 1
    assert out[0]["combined_pred"] is False


def test_get_predictions_includes_latency_when_requested():
    """When include_latency=True, each row has latency_ms."""
    gr = _make_mock_guardrail("gr", return_pass=True)
    data = [{"content": "hi", "label": 0}]
    out = get_predictions(gr, data, include_latency=True)
    assert len(out) == 1
    assert "latency_ms" in out[0]
    assert isinstance(out[0]["latency_ms"], (int, float))
    assert out[0]["latency_ms"] >= 0


def test_get_predictions_label_high_risk_from_label():
    """label_high_risk in output reflects _label_to_bool(label)."""
    gr = _make_mock_guardrail("gr", return_pass=True)
    data = [
        {"content": "a", "label": "high_risk"},
        {"content": "b", "label": "low_risk"},
    ]
    out = get_predictions(gr, data, include_latency=False)
    assert out[0]["label_high_risk"] is True
    assert out[1]["label_high_risk"] is False


def test_get_predictions_content_key_fallback():
    """Content is taken from 'text' or 'prompt' when 'content' missing."""
    gr = _make_mock_guardrail("gr", return_pass=True)
    data = [{"text": "from text", "label": 0}]
    out = get_predictions(gr, data, content_key="content", include_latency=False)
    assert len(out) == 1
    assert out[0]["content"] == "from text"
    gr.evaluate.assert_called_once()
    call_kw = gr.evaluate.call_args[1]
    assert call_kw.get("content") == "from text" or gr.evaluate.call_args[0][0] == "from text"


def test_get_predictions_empty_guardrail_list_returns_empty():
    """Passing empty list of guardrails returns empty list."""
    data = [{"content": "x", "label": 0}]
    out = get_predictions([], data)
    assert out == []


# =============================================================================
# Round-trip: get_predictions -> compute_metrics
# =============================================================================


def test_roundtrip_predictions_then_metrics():
    """Predictions from get_predictions produce correct metrics."""
    # Guardrail that "flags" only the first two contents (return FAIL for them)
    def side_effect(content, **kwargs):
        high_risk = "bad" in content.lower()
        status = GuardrailStatus.FAIL if high_risk else GuardrailStatus.PASS
        return GuardrailResult(status=status)

    gr = Mock()
    gr.config = GuardrailConfig(name="roundtrip_gr", description="", threshold=0.5)
    gr.evaluate = Mock(side_effect=side_effect)

    data = [
        {"content": "bad one", "label": 1},
        {"content": "bad two", "label": 1},
        {"content": "good one", "label": 0},
    ]
    predictions = get_predictions(gr, data, include_latency=False)
    assert len(predictions) == 3
    assert predictions[0]["combined_pred"] is True
    assert predictions[1]["combined_pred"] is True
    assert predictions[2]["combined_pred"] is False

    metrics = compute_metrics_from_predictions(predictions)
    # TP=2, FP=0, FN=0 -> P=1, R=1, F1=1
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.support_high_risk == 2
    assert metrics.support_low_risk == 1
    assert metrics.total_samples == 3
