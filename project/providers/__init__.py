"""LLM Provider implementations"""

from .base import BaseLLMProvider
from .demo_provider import DemoProvider

# All provider exports start with base
__all__ = ["BaseLLMProvider", "DemoProvider"]

# Cohere provider (optional dependency: cohere)
try:
    from .cohere_provider import CohereProvider
    __all__.append("CohereProvider")
except ImportError:
    pass

# OpenAI provider
try:
    from .openai_provider import OpenAIProvider
    __all__.append("OpenAIProvider")
except ImportError:
    pass

# Mistral provider
try:
    from .mistral_provider import MistralProvider
    __all__.append("MistralProvider")
except ImportError:
    pass

