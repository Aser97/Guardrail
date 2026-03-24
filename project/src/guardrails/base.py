"""Base classes for guardrails"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum
import logging


LOGGER = logging.getLogger(__name__)


class GuardrailStatus(Enum):
    """Status of guardrail evaluation"""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class EvaluationType(Enum):
    """Type of content being evaluated"""
    USER_INPUT = "user_input"  # Guardrail for user queries/prompts


@dataclass
class GuardrailResult:
    """Result of a guardrail evaluation"""
    
    status: GuardrailStatus
    score: Optional[float] = None  # 0.0–1.0; convention is guardrail-specific (e.g. classifier: higher = more high_risk/risk)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: Optional[float] = None
    
    @property
    def is_high_risk(self) -> bool:
        """Check if the result is high_risk (status is FAIL or ERROR)."""
        return self.status != GuardrailStatus.PASS
    
    @property
    def is_violation(self) -> bool:
        """Check if there's a violation"""
        return self.status == GuardrailStatus.FAIL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        LOGGER.debug(
            "Serializing GuardrailResult | status=%s score=%s latency_ms=%s metadata_keys=%s",
            self.status.value,
            self.score,
            self.latency_ms,
            list(self.metadata.keys()),
        )
        return {
            "status": self.status.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }


@dataclass
class GuardrailConfig:
    """Configuration for a guardrail"""
    
    name: str
    description: str
    threshold: float = 0.5  # Score threshold for pass/fail
    enabled: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "enabled": self.enabled,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }


@runtime_checkable
class GuardrailProtocol(Protocol):
    """
    Protocol for guardrails: any object with evaluate(content, ...) -> GuardrailResult
    and config can be used in the pipeline (structural subtyping).
    """

    config: GuardrailConfig

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        ...


class BaseGuardrail(ABC):
    """Base class for all guardrails. Subclasses implement evaluate()."""

    def __init__(self, config: GuardrailConfig):
        self.config = config
        LOGGER.info(
            "Initialized guardrail | name=%s threshold=%.4f enabled=%s timeout_seconds=%s",
            config.name,
            config.threshold,
            config.enabled,
            config.timeout_seconds,
        )

    @abstractmethod
    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        """
        Evaluate content against the guardrail.

        Args:
            content: The content to evaluate
            context: Optional context (e.g. user input, conversation history)
            evaluation_type: USER_INPUT

        Returns:
            GuardrailResult with status, score, reasoning, metadata.
        """
        pass
    
    def _create_result(
        self,
        status: GuardrailStatus,
        score: Optional[float] = None,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """Helper method to create a GuardrailResult"""
        LOGGER.debug(
            "Creating guardrail result | guardrail=%s status=%s score=%s reason_present=%s metadata_keys=%s",
            self.config.name,
            status.value,
            score,
            bool(reasoning),
            list((metadata or {}).keys()),
        )
        return GuardrailResult(
            status=status,
            score=score,
            reasoning=reasoning,
            metadata=metadata or {},
        )