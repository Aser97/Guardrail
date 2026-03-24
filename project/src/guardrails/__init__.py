"""Guardrails: flexible, signature-based content safety evaluation.

Any object that implements the guardrail signature (evaluate(content, ...) -> GuardrailResult, plus config) can be used in the pipeline. Use
GuardrailProtocol for typing; subclass BaseGuardrail for shared helpers.

Supports single or stacked guardrails: pass one guardrail or a sequence
(list/tuple); stacks run in order and short-circuit on first failure.
Results are always list-based (input_guardrail_results, output_guardrail_results).

See README.md in this package for full documentation and examples.
"""

from .base import (
    BaseGuardrail,
    GuardrailProtocol,
    GuardrailResult,
    GuardrailConfig,
    GuardrailStatus,
    EvaluationType,
)
from .llm_judge import LLMJudgeGuardrail
from .classifier import ClassifierGuardrail, load_classifier_guardrail
from .metrics import (
    GuardrailMetricsResult,
    get_predictions,
    compute_metrics_from_predictions,
)
from .submission_loader import (
    describe_guardrail,
    load_guardrails_from_module,
    load_evaluation_data,
    write_predictions_csv,
)

# Base exports (always available)
__all__ = [
    # Base classes
    "BaseGuardrail",
    "GuardrailProtocol",
    "GuardrailResult",
    "GuardrailConfig",
    "GuardrailStatus",
    "EvaluationType",
    
    # LLM Judge guardrail
    "LLMJudgeGuardrail",
    # Finetunable classifier guardrail (load with load_classifier_guardrail)
    "ClassifierGuardrail",
    "load_classifier_guardrail",
    # Metrics (precision, recall, F1, latency)
    "GuardrailMetricsResult",
    "get_predictions",
    "compute_metrics_from_predictions",
    "describe_guardrail",
    "load_guardrails_from_module",
    "load_evaluation_data",
    "write_predictions_csv",
]