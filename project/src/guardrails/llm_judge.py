"""LLM-as-Judge guardrail implementation"""

import json
import logging
import os
import re
import time
from typing import Optional, Dict, Any

from .base import BaseGuardrail, GuardrailResult, GuardrailConfig, GuardrailStatus, EvaluationType
from providers.base import BaseLLMProvider, LLMMessage
from ..prompt_templates.guardrail_prompt_template import DEFAULT_USER_INPUT_PROMPT

LOGGER = logging.getLogger(__name__)


def _env_set(name: str) -> bool:
    """Return whether an env var is set without exposing secret values."""
    return bool(os.getenv(name))


class LLMJudgeGuardrail(BaseGuardrail):
    """
    LLM-as-Judge guardrail that uses an LLM to evaluate content

    Evaluates user input content before processing.
    """
    
    def __init__(
        self,
        config: GuardrailConfig,
        llm_provider: BaseLLMProvider,
        user_input_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        response_format: str = "json"
    ):
        """
        Initialize LLM Judge guardrail for user input evaluation.
        
        Args:
            config: Guardrail configuration
            llm_provider: LLM provider for evaluations
            user_input_prompt: Custom system prompt for evaluating user inputs (optional)
            system_prompt: Alias for user_input_prompt (backward compatibility)
            response_format: Expected response format (default: "json")
        """
        super().__init__(config)
        self.llm_provider = llm_provider

        self.user_input_prompt = system_prompt or user_input_prompt or DEFAULT_USER_INPUT_PROMPT
        self.response_format = response_format
        provider_name = type(llm_provider).__name__
        LOGGER.info(
            "Initialized LLMJudgeGuardrail | name=%s provider=%s threshold=%.4f max_retries=%d response_format=%s",
            self.config.name,
            provider_name,
            self.config.threshold,
            self.config.max_retries,
            self.response_format,
        )
        LOGGER.info(
            "LLM env presence | BUZZ_COHERE_AUTH_TOKEN_set=%s BUZZ_GPT_OSS_AUTH_TOKEN_set=%s BUZZ_MISTRAL_LARGE_AUTH_TOKEN_set=%s BUZZ_COHERE_API_set=%s BUZZ_GPT_OSS_API_set=%s BUZZ_MISTRAL_LARGE_API_set=%s",
            _env_set("BUZZ_COHERE_AUTH_TOKEN"),
            _env_set("BUZZ_GPT_OSS_AUTH_TOKEN"),
            _env_set("BUZZ_MISTRAL_LARGE_AUTH_TOKEN"),
            _env_set("BUZZ_COHERE_API"),
            _env_set("BUZZ_GPT_OSS_API"),
            _env_set("BUZZ_MISTRAL_LARGE_API"),
        )
    
    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        """
        Evaluate content using LLM as judge.

        Args:
            content: The content to evaluate
            context: Optional context information
            evaluation_type: USER_INPUT

        Returns:
            GuardrailResult with evaluation details
        """
        start_time = time.time()
        evaluation_type_value = (
            evaluation_type.value if isinstance(evaluation_type, EvaluationType) else str(evaluation_type)
        )
        try:
            if not isinstance(content, str):
                LOGGER.error(
                    "Input dataset schema not respected in llm_judge evaluate: expected 'content' as str, got %s",
                    type(content).__name__,
                )
                raise ValueError(
                    "Input dataset schema not respected: each row must provide text content as a string."
                )
            if context is not None and not isinstance(context, dict):
                LOGGER.error(
                    "Input dataset schema not respected in llm_judge evaluate: expected 'context' as dict or None, got %s",
                    type(context).__name__,
                )
                raise ValueError(
                    "Input dataset schema not respected: context must be a dictionary when provided."
                )
            if not isinstance(evaluation_type, EvaluationType):
                LOGGER.error(
                    "Input dataset schema not respected in llm_judge evaluate: expected 'evaluation_type' as EvaluationType, got %s",
                    type(evaluation_type).__name__,
                )
                raise ValueError(
                    "Input dataset schema not respected: evaluation_type must be a valid EvaluationType."
                )
            content_preview = (content or "").strip().replace("\n", " ")[:120]
            LOGGER.debug(
                "LLM judge evaluate start | guardrail=%s evaluation_type=%s content_chars=%d context_keys=%s preview=%r",
                self.config.name,
                evaluation_type.value,
                len((content or "").strip()),
                sorted((context or {}).keys()),
                content_preview,
            )
            messages = [
                LLMMessage(role="system", content=self.user_input_prompt),
                LLMMessage(
                    role="user",
                    content=self._format_evaluation_prompt(content, context),
                ),
            ]
            response = self._generate_with_retry(messages)
            
            # Parse response
            evaluation = self._parse_llm_response(response.content)
            
            # Determine status based on evaluation
            status = self._determine_status(evaluation)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.info(
                "LLM judge evaluate success | guardrail=%s provider=%s status=%s score=%s latency_ms=%.3f model=%s",
                self.config.name,
                type(self.llm_provider).__name__,
                status.value,
                evaluation.get("score"),
                latency_ms,
                getattr(response, "model", None),
            )
            
            return GuardrailResult(
                status=status,
                score=evaluation.get("score"),
                metadata={
                    "evaluation_type": evaluation_type_value,
                    "llm_response": response.content,
                    "model": response.model,
                    "usage": response.usage,
                },
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            is_parse_error = isinstance(e, ValueError) and "parse" in str(e).lower()
            fail_open = bool(getattr(self.config, "fail_open", False))
            if fail_open:
                LOGGER.exception(
                    "LLM judge evaluate failed | guardrail=%s provider=%s fail_open=%s resolved_status=%s is_parse_error=%s latency_ms=%.3f",
                    self.config.name,
                    type(self.llm_provider).__name__,
                    fail_open,
                    GuardrailStatus.PASS.value,
                    is_parse_error,
                    latency_ms,
                )
                return GuardrailResult(
                    status=GuardrailStatus.PASS,
                    reasoning=f"Error during evaluation: {str(e)}",
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "is_parse_error": is_parse_error,
                        "evaluation_type": evaluation_type_value,
                        "fail_open": fail_open,
                    },
                    latency_ms=latency_ms,
                )

            # fail_open=False: log and fail fast so evaluator does not silently continue.
            LOGGER.exception(
                "LLM judge evaluate failed and will raise | guardrail=%s provider=%s fail_open=%s latency_ms=%.3f",
                self.config.name,
                type(self.llm_provider).__name__,
                fail_open,
                latency_ms,
            )
            raise RuntimeError(
                f"LLM judge evaluation failed for guardrail '{self.config.name}': {e}"
            ) from e

    def _format_evaluation_prompt(
        self, 
        content: str, 
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Format the evaluation prompt for user input."""
        prompt = f"Evaluate the following user input:\n\n{content}"
        
        if context:
            prompt += f"\n\nContext:\n{json.dumps(context, indent=2)}"
            LOGGER.debug("Evaluation prompt includes context keys: %s", sorted(context.keys()))
        
        prompt += "\n\nProvide your evaluation in JSON format."
        LOGGER.debug("Evaluation prompt length=%d chars", len(prompt))
        
        return prompt
    
    def _generate_with_retry(self, messages):
        """Generate with retry logic (sync)."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                LOGGER.debug(
                    "LLM generate attempt %d/%d | provider=%s",
                    attempt + 1,
                    self.config.max_retries,
                    type(self.llm_provider).__name__,
                )
                return self.llm_provider.generate_sync(messages)
            except Exception as e:
                last_error = e
                wait_seconds = 2**attempt if attempt < self.config.max_retries - 1 else 0
                LOGGER.warning(
                    "LLM generate attempt failed | attempt=%d/%d error_type=%s error=%s next_backoff_seconds=%s",
                    attempt + 1,
                    self.config.max_retries,
                    type(e).__name__,
                    str(e),
                    wait_seconds if wait_seconds else "<none>",
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
        LOGGER.error(
            "Exhausted all LLM generate retries | provider=%s max_retries=%d",
            type(self.llm_provider).__name__,
            self.config.max_retries,
        )
        raise last_error
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate the LLM response
        
        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        try:
            text = (response_text or "").strip()
            LOGGER.debug("Parsing LLM response text length=%d", len(text))

            # 1) Raw JSON attempt.
            candidates = [text] if text else []

            # 2) JSON inside fenced code blocks, then broad {...} slice.
            fenced_blocks = re.findall(
                r"```(?:json)?\s*([\s\S]*?)\s*```",
                text,
                flags=re.IGNORECASE,
            )
            candidates.extend(block.strip() for block in fenced_blocks if block.strip())

            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx].strip()
                if json_str:
                    LOGGER.debug(
                        "Extracted JSON substring from LLM response | start_idx=%d end_idx=%d substring_len=%d",
                        start_idx,
                        end_idx,
                        len(json_str),
                    )
                    candidates.append(json_str)

            # Preserve order while removing duplicate candidates.
            candidates = list(dict.fromkeys(candidates))
            for idx, candidate in enumerate(candidates):
                try:
                    parsed = json.loads(candidate)
                    LOGGER.debug("LLM JSON parse succeeded via candidate index=%d", idx)
                    normalized = self._validate_and_normalize_response(parsed)
                    LOGGER.debug(
                        "Parsed and normalized LLM response | high_risk=%s score=%s",
                        normalized.get("high_risk"),
                        normalized.get("score"),
                    )
                    return normalized
                except Exception as e:
                    LOGGER.debug("Candidate parse failed | index=%d error=%s", idx, str(e))

            # 3) Last resort: extract required keys from malformed text.
            parsed = self._extract_required_fields_from_text(text)
            normalized = self._validate_and_normalize_response(parsed)
            LOGGER.debug(
                "Regex fallback parsed and normalized LLM response | high_risk=%s score=%s",
                normalized.get("high_risk"),
                normalized.get("score"),
            )
            return normalized
        except Exception as e:
            raise ValueError(f"Unable to parse LLM response as JSON: {str(e)}")

    def _extract_required_fields_from_text(self, response_text: str) -> Dict[str, Any]:
        """Best-effort extraction of required fields from malformed outputs."""
        high_risk_match = re.search(
            r"""["']?high_risk["']?\s*[:=]\s*(true|false)""",
            response_text,
            flags=re.IGNORECASE,
        )
        low_risk_match = re.search(
            r"""["']?low_risk["']?\s*[:=]\s*(true|false)""",
            response_text,
            flags=re.IGNORECASE,
        )
        score_match = re.search(
            r"""["']?score["']?\s*[:=]\s*["']?(-?\d+(?:\.\d+)?)["']?""",
            response_text,
            flags=re.IGNORECASE,
        )
        if not high_risk_match and not low_risk_match:
            raise ValueError("Could not extract required field 'high_risk' or 'low_risk'")
        if not score_match:
            raise ValueError("Could not extract required field 'score'")

        if high_risk_match:
            high_risk = high_risk_match.group(1).lower() == "true"
        else:
            low_risk = low_risk_match.group(1).lower() == "true"
            high_risk = not low_risk

        return {
            "high_risk": high_risk,
            "score": float(score_match.group(1)),
        }
    
    def _validate_and_normalize_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the parsed response to ensure expected keys exist
        
        Args:
            parsed: The parsed JSON response
            
        Returns:
            Normalized response with all expected keys
            
        Raises:
            ValueError: If critical keys are missing or invalid
        """
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid response type: expected object, got {type(parsed).__name__}")

        LOGGER.debug("Validating parsed LLM response keys: %s", sorted(parsed.keys()))
        has_high_risk = "high_risk" in parsed
        has_low_risk = "low_risk" in parsed
        if not has_high_risk and not has_low_risk:
            raise ValueError("Missing required key: need 'high_risk' or 'low_risk' in LLM response")
        if "score" not in parsed:
            raise ValueError("Missing required key 'score' in LLM response")

        if has_high_risk and not isinstance(parsed["high_risk"], bool):
            raise ValueError(f"Invalid type for 'high_risk': expected bool, got {type(parsed['high_risk']).__name__}")
        if has_low_risk and not isinstance(parsed["low_risk"], bool):
            raise ValueError(f"Invalid type for 'low_risk': expected bool, got {type(parsed['low_risk']).__name__}")
        if not isinstance(parsed["score"], (int, float)):
            raise ValueError(f"Invalid type for 'score': expected number, got {type(parsed['score']).__name__}")

        if has_high_risk:
            high_risk = parsed["high_risk"]
        else:
            high_risk = not parsed["low_risk"]

        score = float(parsed["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score {score} is out of valid range [0.0, 1.0]")

        normalized = {
            "high_risk": high_risk,
            "low_risk": not high_risk,
            "score": score,
        }
        LOGGER.debug(
            "Normalized LLM response | high_risk=%s score=%.6f",
            normalized["high_risk"],
            normalized["score"],
        )
        
        return normalized
    
    def _determine_status(self, evaluation: Dict[str, Any]) -> GuardrailStatus:
        """Determine guardrail status from evaluation"""
        score = evaluation["score"]
        is_high_risk = evaluation["high_risk"]
        LOGGER.debug(
            "Determining guardrail status | high_risk=%s score=%s threshold=%.4f",
            is_high_risk,
            score,
            self.config.threshold,
        )
        
        # If explicitly marked as high_risk
        if is_high_risk:
            LOGGER.debug("Status decision: FAIL (explicit high_risk=true)")
            return GuardrailStatus.FAIL
        
        # Use threshold to determine status (higher score = more dangerous)
        if score >= self.config.threshold:
            LOGGER.debug("Status decision: FAIL (score above threshold)")
            return GuardrailStatus.FAIL
        else:
            LOGGER.debug("Status decision: PASS (score below threshold and not explicit high_risk)")
            return GuardrailStatus.PASS

