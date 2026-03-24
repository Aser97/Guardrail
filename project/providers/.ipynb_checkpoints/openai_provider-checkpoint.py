"""OpenAI provider implementation"""

import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

from .base import BaseLLMProvider, LLMMessage, LLMResponse

load_dotenv()

DEFAULT_BUZZ_GPT_OSS_API = "https://gpt-oss-120b-3e1ds.inference.buzzperformancecloud.com"


class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI-compatible chat-completions API.
    Sends requests to /v1/chat/completions with "messages".
    Base URL defaults to BUZZ GPT OSS endpoint; override with BUZZ_GPT_OSS_API or base_url.
    API key via BUZZ_GPT_OSS_AUTH_TOKEN or api_key.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1000,
        verify_ssl: bool = False,
        timeout_s: float = 60.0,
        **kwargs
    ):
        base_url = (
            base_url or os.getenv("BUZZ_GPT_OSS_API") or DEFAULT_BUZZ_GPT_OSS_API
        ).strip().rstrip("/")
        api_key = api_key or os.getenv("BUZZ_GPT_OSS_AUTH_TOKEN")
        if not api_key:
            raise ValueError(
                "OpenAI api_key must be set via the api_key argument or BUZZ_GPT_OSS_AUTH_TOKEN env var"
            )

        super().__init__(api_key, model, temperature, max_tokens, **kwargs)

        self.base_url = base_url
        self.endpoint_url = f"{self.base_url}/v1/chat/completions"
        self.system_prompt = system_prompt or ""

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(timeout_s, connect=10.0)
        self.sync_client = httpx.Client(headers=headers, verify=verify_ssl, timeout=timeout)
        self.async_client = httpx.AsyncClient(headers=headers, verify=verify_ssl, timeout=timeout)

    # Optional but recommended to avoid resource leaks
    def close(self) -> None:
        self.sync_client.close()

    async def aclose(self) -> None:
        await self.async_client.aclose()

    def _messages_to_api(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Build chat API messages list; prepend provider system_prompt if set."""
        api = [{"role": msg.role, "content": msg.content or ""} for msg in messages]
        if self.system_prompt and (not api or api[0].get("role") != "system"):
            api = [{"role": "system", "content": self.system_prompt}] + api
        return api

    def _prepare_request_payload(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_to_api(messages),
            "max_tokens": self._get_max_tokens(max_tokens),
        }
        temp_value = self._get_temperature(temperature)
        if temp_value is not None:
            payload["temperature"] = temp_value
        payload.update(self.extra_params)
        payload.update(kwargs)
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        if "error" in data:
            raise RuntimeError(f"LLM error: {data['error']}")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected response format (no choices): {data}")

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()

        usage = data.get("usage")
        if usage:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            metadata={"response_id": data.get("id")},
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        payload = self._prepare_request_payload(messages, temperature, max_tokens, **kwargs)

        resp = await self.async_client.post(self.endpoint_url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

        return self._parse_response(resp.json())

    def generate_sync(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        payload = self._prepare_request_payload(messages, temperature, max_tokens, **kwargs)

        resp = self.sync_client.post(self.endpoint_url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

        return self._parse_response(resp.json())
