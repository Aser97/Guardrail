"""
scripts/llm_client.py
API clients for all models used in the pipeline.

Two clients:

  LLMClient       — Hackathon models (Buzz Performance Cloud, OpenAI-compatible)
                    Each model has its own base URL and auth token (per .env).
                    Mistral Large 3 : BUZZ_MISTRAL_LARGE_API / BUZZ_MISTRAL_LARGE_AUTH_TOKEN
                    C4AI Command A  : BUZZ_COHERE_API        / BUZZ_COHERE_AUTH_TOKEN
                    GPT-OSS         : BUZZ_GPT_OSS_API       / BUZZ_GPT_OSS_AUTH_TOKEN

  AnthropicClient — Anthropic API (api.anthropic.com)
                    Claude 3.7 Sonnet — used exclusively as PAIR judge (Phase 2g)
                    Auth: ANTHROPIC_API_KEY (x-api-key)

Usage:
    from llm_client import LLMClient, AnthropicClient
    from llm_client import MISTRAL, COMMAND, CLAUDE_SONNET

    hackathon = LLMClient()
    claude    = AnthropicClient()

    text  = hackathon.complete(MISTRAL, messages)
    judge = claude.complete(CLAUDE_SONNET, messages)
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model identifiers — must match what the Buzz endpoints expect exactly
# (from .env: MISTRAL_MODEL / COHERE_MODEL / OPENAI_MODEL)
# ──────────────────────────────────────────────────────────────────────────────

MISTRAL      = "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4"
COMMAND      = "CohereLabs/c4ai-command-a-03-2025"
GPT_OSS      = "openai/gpt-oss-120b"
HACKATHON_MODELS = [MISTRAL, COMMAND, GPT_OSS]

# Anthropic models
CLAUDE_SONNET = "claude-sonnet-4-6"  # PAIR judge (Phase 2g) — private key only

# ──────────────────────────────────────────────────────────────────────────────
# Private models (Phase 1b, 2f, 2g) — two providers, one router (PrivateClient)
#
#   PRIVATE_MISTRAL → Mistral AI own API (console.mistral.ai)
#                     model string "mistral-large-latest" maps to their current Large 3
#                     env var: MISTRAL_API_KEY
#
#   PRIVATE_SUPPORT → Together AI (together.ai)
#                     confirmed available on Together AI dashboard
#                     env var: TOGETHER_API_KEY
# ──────────────────────────────────────────────────────────────────────────────
PRIVATE_MISTRAL = "mistral-large-latest"                      # Mistral AI API
PRIVATE_SUPPORT = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Together AI — serverless
PRIVATE_HAIKU   = "claude-haiku-4-5-20251001"                 # Anthropic API — augmentation (Phase 2a-2d)

# ──────────────────────────────────────────────────────────────────────────────
# Per-model env var names (matching the organizers' .env exactly)
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_CONFIG: dict[str, dict[str, str]] = {
    MISTRAL: {
        "base_env":  "BUZZ_MISTRAL_LARGE_API",
        "token_env": "BUZZ_MISTRAL_LARGE_AUTH_TOKEN",
        "default_base": "https://mistral-large-3-675b-instruct-nvfp4-3e1ds.inference.buzzperformancecloud.com",
    },
    COMMAND: {
        "base_env":  "BUZZ_COHERE_API",
        "token_env": "BUZZ_COHERE_AUTH_TOKEN",
        "default_base": "https://cohere-c4ai-3e1ds.inference.buzzperformancecloud.com",
    },
    GPT_OSS: {
        "base_env":  "BUZZ_GPT_OSS_API",
        "token_env": "BUZZ_GPT_OSS_AUTH_TOKEN",
        "default_base": "https://gpt-oss-120b-3e1ds.inference.buzzperformancecloud.com",
    },
}

# Anthropic endpoint
_ANTHROPIC_API_KEY_ENV  = "ANTHROPIC_API_KEY"
_ANTHROPIC_API_BASE     = "https://api.anthropic.com/v1"
_ANTHROPIC_API_VERSION  = "2023-06-01"


def _resolve_model_endpoint(model: str) -> tuple[str, str]:
    """Return (base_url, token) for a given hackathon model, reading from env."""
    cfg = _MODEL_CONFIG.get(model)
    if cfg is None:
        raise ValueError(
            f"Unknown hackathon model: {model!r}. "
            f"Known models: {list(_MODEL_CONFIG)}"
        )
    base  = os.getenv(cfg["base_env"], cfg["default_base"]).strip().rstrip("/")
    token = os.getenv(cfg["token_env"], "")
    if not token:
        LOGGER.warning(
            "No token found for model %s. Set %s in your .env file.",
            model, cfg["token_env"],
        )
    return base, token


class LLMClient:
    """
    Thin wrapper around the Buzz Performance Cloud OpenAI-compatible REST API.

    Each hackathon model has its own base URL and auth token — these are read
    from the per-model env vars defined in _MODEL_CONFIG above, which match
    the organizers' .env file exactly.

    Parameters
    ----------
    max_retries : int
        Number of retry attempts on transient errors (default 4).
    retry_delay : float
        Base delay in seconds between retries (exponential back-off).
    """

    def __init__(
        self,
        max_retries: int = 4,
        retry_delay: float = 2.0,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Lazy import — openai may not be installed in every env
        try:
            import openai  # noqa: F401
            self._use_openai_sdk = True
        except ImportError:
            self._use_openai_sdk = False
            LOGGER.info("openai package not available; falling back to httpx.")

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def complete(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stop: Optional[list[str]] = None,
        timeout: int = 90,
    ) -> str:
        """
        Call the chat-completion endpoint for the given model and return the
        assistant's text. Routes to the correct Buzz endpoint automatically.

        Raises
        ------
        RuntimeError
            After max_retries exhausted.
        """
        api_base, api_token = _resolve_model_endpoint(model)

        for attempt in range(self.max_retries):
            try:
                if self._use_openai_sdk:
                    return self._complete_openai(
                        model, messages, api_base=api_base, api_token=api_token,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=top_p, stop=stop, timeout=timeout,
                    )
                else:
                    return self._complete_httpx(
                        model, messages, api_base=api_base, api_token=api_token,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=top_p, stop=stop, timeout=timeout,
                    )
            except Exception as exc:
                wait = self.retry_delay * (2 ** attempt)
                LOGGER.warning(
                    "LLM call failed (attempt %d/%d) model=%s error=%s — retrying in %.1fs",
                    attempt + 1, self.max_retries, model, exc, wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"All {self.max_retries} attempts failed for model={model}: {exc}"
                    ) from exc

        raise RuntimeError("Unreachable")  # pragma: no cover

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _complete_openai(
        self,
        model: str,
        messages: list[dict],
        *,
        api_base: str,
        api_token: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[list[str]],
        timeout: int,
    ) -> str:
        import httpx
        from openai import OpenAI

        # Buzz endpoints use self-signed certs — disable SSL verification
        http_client = httpx.Client(verify=False, timeout=timeout)
        client = OpenAI(
            api_key=api_token,
            base_url=f"{api_base}/v1",
            timeout=timeout,
            http_client=http_client,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        return resp.choices[0].message.content or ""

    def _complete_httpx(
        self,
        model: str,
        messages: list[dict],
        *,
        api_base: str,
        api_token: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[list[str]],
        timeout: int,
    ) -> str:
        try:
            import httpx
        except ImportError:
            return self._complete_urllib(
                model, messages, api_base=api_base, api_token=api_token,
                temperature=temperature, max_tokens=max_tokens,
                stop=stop, timeout=timeout,
            )

        url = f"{api_base}/v1/chat/completions"
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop

        # verify=False: Buzz Performance Cloud uses self-signed certificates
        with httpx.Client(timeout=timeout, verify=False) as http:
            r = http.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_token}"},
            )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _complete_urllib(
        self,
        model: str,
        messages: list[dict],
        *,
        api_base: str,
        api_token: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        timeout: int,
    ) -> str:
        import urllib.request
        import ssl
        import json as _json

        url = f"{api_base}/v1/chat/completions"
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        req = urllib.request.Request(
            url,
            data=_json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # Skip SSL verification for Buzz self-signed certs
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = _json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic client  (PAIR judge only — Phase 2f)
# ──────────────────────────────────────────────────────────────────────────────

class AnthropicClient:
    """
    Thin client for the Anthropic Messages API.

    Used exclusively for the PAIR adversarial judge (Claude 3.7 Sonnet).
    Requires ANTHROPIC_API_KEY set in the environment.
    This is a separate paid account from Claude.ai Pro — sign up at
    console.anthropic.com and add a credit card for pay-as-you-go billing.

    Exposes the same .complete(model, messages, ...) interface as LLMClient
    so generate_pair.py can call both clients uniformly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 4,
        retry_delay: float = 2.0,
    ):
        self.api_key     = api_key or os.getenv(_ANTHROPIC_API_KEY_ENV, "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            LOGGER.warning(
                "No Anthropic API key found. Set %s before running PAIR generation.",
                _ANTHROPIC_API_KEY_ENV,
            )

        try:
            import anthropic  # noqa: F401
            self._use_anthropic_sdk = True
        except ImportError:
            self._use_anthropic_sdk = False
            LOGGER.info("anthropic package not available; falling back to httpx.")

    def complete(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 90,
    ) -> str:
        """
        Call the Anthropic Messages API and return the assistant's text.
        system prompt must be passed as the first message with role="system"
        (same convention as LLMClient).
        """
        for attempt in range(self.max_retries):
            try:
                if self._use_anthropic_sdk:
                    return self._complete_sdk(
                        model, messages, temperature=temperature,
                        max_tokens=max_tokens, timeout=timeout,
                    )
                else:
                    return self._complete_httpx(
                        model, messages, temperature=temperature,
                        max_tokens=max_tokens, timeout=timeout,
                    )
            except Exception as exc:
                wait = self.retry_delay * (2 ** attempt)
                LOGGER.warning(
                    "Anthropic call failed (attempt %d/%d) model=%s error=%s — retrying in %.1fs",
                    attempt + 1, self.max_retries, model, exc, wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"All {self.max_retries} Anthropic attempts failed for model={model}: {exc}"
                    ) from exc

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _complete_sdk(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        import anthropic

        # Anthropic SDK separates system prompt from the messages list
        system = ""
        filtered: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout)
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=filtered,
        )
        if system:
            kwargs["system"] = system

        resp = client.messages.create(**kwargs)
        return resp.content[0].text

    def _complete_httpx(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        try:
            import httpx
        except ImportError:
            raise RuntimeError(
                "Neither 'anthropic' nor 'httpx' is installed. "
                "Run: pip install anthropic"
            )

        # Separate system prompt
        system = ""
        filtered: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        payload: dict = {
            "model":      model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages":   filtered,
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key":          self.api_key,
            "anthropic-version":  _ANTHROPIC_API_VERSION,
            "content-type":       "application/json",
        }

        with httpx.Client(timeout=timeout) as http:
            r = http.post(
                f"{_ANTHROPIC_API_BASE}/messages",
                json=payload,
                headers=headers,
            )
        r.raise_for_status()
        return r.json()["content"][0]["text"]


# ──────────────────────────────────────────────────────────────────────────────
# Together AI client  (Phase 1b + 2f — private key)
# ──────────────────────────────────────────────────────────────────────────────

_TOGETHER_BASE      = "https://api.together.xyz"
_TOGETHER_KEY_ENV   = "TOGETHER_API_KEY"

_MISTRAL_AI_BASE    = "https://api.mistral.ai"
_MISTRAL_AI_KEY_ENV = "MISTRAL_API_KEY"


class TogetherClient:
    """
    Generic OpenAI-compatible client for private API providers.

    Defaults to Together AI, but can target any OpenAI-compatible endpoint
    by passing base_url and api_key_env at construction.

    Used directly via PrivateClient (which routes to Mistral AI or Together AI
    based on the model requested).
    """

    def __init__(
        self,
        base_url: str = _TOGETHER_BASE,
        api_key_env: str = _TOGETHER_KEY_ENV,
        max_retries: int = 4,
        retry_delay: float = 2.0,
    ):
        self.base_url    = base_url.rstrip("/")
        self.api_key     = os.getenv(api_key_env, "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            LOGGER.warning(
                "No API key found for %s. Set %s in your .env file.",
                base_url, api_key_env,
            )

        try:
            import openai  # noqa: F401
            self._use_openai_sdk = True
        except ImportError:
            self._use_openai_sdk = False
            LOGGER.info("openai package not available; falling back to httpx.")

    def complete(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stop: Optional[list[str]] = None,
        timeout: int = 90,
    ) -> str:
        """Call Together AI and return the assistant's text."""
        for attempt in range(self.max_retries):
            try:
                if self._use_openai_sdk:
                    return self._complete_openai(
                        model, messages,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=top_p, stop=stop, timeout=timeout,
                    )
                else:
                    return self._complete_httpx(
                        model, messages,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=top_p, stop=stop, timeout=timeout,
                    )
            except Exception as exc:
                wait = self.retry_delay * (2 ** attempt)
                LOGGER.warning(
                    "Together call failed (attempt %d/%d) model=%s error=%s — retrying in %.1fs",
                    attempt + 1, self.max_retries, model, exc, wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"All {self.max_retries} Together attempts failed for model={model}: {exc}"
                    ) from exc

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _complete_openai(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[list[str]],
        timeout: int,
    ) -> str:
        from openai import OpenAI

        # Private providers use proper SSL — no verify=False needed
        client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.base_url}/v1",
            timeout=timeout,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        return resp.choices[0].message.content or ""

    def _complete_httpx(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[list[str]],
        timeout: int,
    ) -> str:
        try:
            import httpx
        except ImportError:
            raise RuntimeError(
                "Neither 'openai' nor 'httpx' is installed. Run: pip install openai"
            )

        url = f"{self.base_url}/v1/chat/completions"
        payload: dict = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "top_p":       top_p,
        }
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=timeout) as http:
            r = http.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
# PrivateClient — routes PRIVATE_MISTRAL → Mistral AI, PRIVATE_SUPPORT → Together AI
# Used by generate_camel.py (Phase 1b), evolve_conversations.py (Phase 2f),
# and generate_pair.py generator (Phase 2g).
# ──────────────────────────────────────────────────────────────────────────────

class PrivateClient:
    """
    Router that dispatches to the correct private provider based on model.

      PRIVATE_MISTRAL → Mistral AI (api.mistral.ai, MISTRAL_API_KEY)
      everything else → Together AI (api.together.xyz, TOGETHER_API_KEY)
    """

    def __init__(self, max_retries: int = 4, retry_delay: float = 2.0):
        self._mistral = TogetherClient(
            base_url=_MISTRAL_AI_BASE,
            api_key_env=_MISTRAL_AI_KEY_ENV,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._together = TogetherClient(
            base_url=_TOGETHER_BASE,
            api_key_env=_TOGETHER_KEY_ENV,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._anthropic = AnthropicClient(
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def complete(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stop: Optional[list[str]] = None,
        timeout: int = 90,
    ) -> str:
        if model == PRIVATE_MISTRAL:
            return self._mistral.complete(
                model, messages,
                temperature=temperature, max_tokens=max_tokens,
                top_p=top_p, stop=stop, timeout=timeout,
            )
        elif model == PRIVATE_HAIKU:
            # AnthropicClient does not use top_p / stop
            return self._anthropic.complete(
                model, messages,
                temperature=temperature, max_tokens=max_tokens, timeout=timeout,
            )
        else:
            return self._together.complete(
                model, messages,
                temperature=temperature, max_tokens=max_tokens,
                top_p=top_p, stop=stop, timeout=timeout,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singletons (import and use directly)
# ──────────────────────────────────────────────────────────────────────────────
_client: Optional[LLMClient] = None
_anthropic_client: Optional[AnthropicClient] = None


def get_client() -> LLMClient:
    """Return (and lazily create) the shared hackathon LLMClient."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def get_anthropic_client() -> AnthropicClient:
    """Return (and lazily create) the shared AnthropicClient (PAIR judge)."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AnthropicClient()
    return _anthropic_client


_together_client: Optional[TogetherClient] = None
_private_client:  Optional[PrivateClient]  = None


def get_together_client() -> TogetherClient:
    """Return (and lazily create) the shared TogetherClient (Together AI only)."""
    global _together_client
    if _together_client is None:
        _together_client = TogetherClient()
    return _together_client


def get_private_client() -> PrivateClient:
    """Return (and lazily create) the shared PrivateClient (Phase 1b, 2f, 2g generator)."""
    global _private_client
    if _private_client is None:
        _private_client = PrivateClient()
    return _private_client


def quick_complete(
    model: str,
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.75,
    max_tokens: int = 1024,
) -> str:
    """
    Convenience wrapper: single user turn → assistant reply.
    Routes to the correct client based on model name.

    Parameters
    ----------
    model : str
        One of MISTRAL, COMMAND, GPT_OSS (hackathon) or CLAUDE_SONNET (Anthropic PAIR judge).
    """
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    if model == CLAUDE_SONNET:
        return get_anthropic_client().complete(
            model, msgs, temperature=temperature, max_tokens=max_tokens,
        )
    return get_client().complete(
        model, msgs, temperature=temperature, max_tokens=max_tokens,
    )
