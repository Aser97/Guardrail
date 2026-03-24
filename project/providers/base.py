"""Base LLM provider interface"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class LLMMessage:
    """Represents a message in LLM conversation"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM
        
        Args:
            messages: List of messages in the conversation
            temperature: Optional override for temperature
            max_tokens: Optional override for max tokens
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    def generate_sync(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Synchronous version of generate"""
        pass
    
    def _get_temperature(self, override: Optional[float]) -> Optional[float]:
        """Get temperature with optional override. Returns None to use model default."""
        return override if override is not None else self.temperature
    
    def _get_max_tokens(self, override: Optional[int]) -> int:
        """Get max_tokens with optional override"""
        return override if override is not None else self.max_tokens

