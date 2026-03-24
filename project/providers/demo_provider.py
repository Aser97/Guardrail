"""Demo provider for testing and demonstrations without requiring API keys"""

import json
from typing import List, Optional

from .base import BaseLLMProvider, LLMMessage, LLMResponse


class DemoProvider(BaseLLMProvider):
    """
    Demo provider that simulates realistic LLM responses for testing and demonstrations.
    
    This provider does not require an API key and generates responses based on
    keyword analysis. It's useful for:
    - Testing guardrail functionality without API costs
    - Demonstrations and tutorials
    - Development and debugging
    
    Note: This provider provides simulated responses and should not be used in production.
    """
    
    def __init__(
        self,
        model: str = "demo-model",
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize demo provider.
        
        Args:
            model: Model name (for display purposes)
            temperature: Temperature parameter (not used, kept for compatibility)
            max_tokens: Max tokens (not used, kept for compatibility)
            **kwargs: Additional parameters (not used, kept for compatibility)
        """
        # Don't call super().__init__() since we don't need api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
        
        # Risk level keywords for analysis
        self.high_risk = ["kill myself", "end my life", "suicide", "want to die"]
        self.medium_risk = ["can't go on", "no point", "hurt myself"]
        self.low_risk = ["struggling", "overwhelmed", "depressed"]
    
    def _analyze(self, content: str) -> dict:
        """
        Analyze content and return a simulated risk assessment.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with risk assessment including:
            - low_risk: Boolean indicating if content is low_risk
            - score: Risk score (0-1, higher is more high_risk; aligns with classifier: high = block)
            - risk_level: Risk level (none/low/medium/high)
            - reasoning: Explanation of the assessment
            - violations: List of detected violations
            - indicators: List of detected risk indicators
        """
        # Score convention: risk score 0–1 (higher = more high_risk). Aligns with classifier guardrail: high score = block.
        lower = content.lower()

        if any(kw in lower for kw in self.high_risk):
            return {
                "low_risk": False,
                "score": 0.85,
                "risk_level": "high",
                "reasoning": "Explicit self-harm or suicide ideation",
                "violations": ["suicide ideation"],
                "indicators": [kw for kw in self.high_risk if kw in lower]
            }
        elif any(kw in lower for kw in self.medium_risk):
            return {
                "low_risk": False,
                "score": 0.55,
                "risk_level": "medium",
                "reasoning": "Concerning indicators of potential self-harm",
                "violations": ["concerning language"],
                "indicators": [kw for kw in self.medium_risk if kw in lower]
            }
        elif any(kw in lower for kw in self.low_risk):
            return {
                "low_risk": True,
                "score": 0.35,
                "risk_level": "low",
                "reasoning": "Distress but no clear self-harm intent",
                "violations": [],
                "indicators": []
            }
        else:
            return {
                "low_risk": True,
                "score": 0.08,
                "risk_level": "none",
                "reasoning": "Normal emotions without self-harm indicators",
                "violations": [],
                "indicators": []
            }
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a simulated response (async version).
        
        Args:
            messages: List of messages in the conversation
            temperature: Temperature parameter (not used)
            max_tokens: Max tokens parameter (not used)
            **kwargs: Additional parameters (not used)
            
        Returns:
            LLMResponse with simulated analysis
        """
        return self.generate_sync(messages, temperature, max_tokens, **kwargs)
    
    def generate_sync(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a simulated response (sync version).
        
        Args:
            messages: List of messages in the conversation
            temperature: Temperature parameter (not used)
            max_tokens: Max tokens parameter (not used)
            **kwargs: Additional parameters (not used)
            
        Returns:
            LLMResponse with simulated analysis
        """
        # Extract content from the last message
        content = messages[-1].content if messages else ""
        
        # Analyze the content
        analysis = self._analyze(content)
        
        # Return as JSON-formatted response
        return LLMResponse(
            content=json.dumps(analysis, indent=2),
            model=self.model,
            usage={"total_tokens": 150},
            finish_reason="stop",
            metadata={"provider": "demo"}
        )

