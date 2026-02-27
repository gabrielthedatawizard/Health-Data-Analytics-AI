from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.llm_gate import SchemaValidationError, validate_schema


class LLMCircuitOpenError(RuntimeError):
    pass


class LLMClient(ABC):
    @abstractmethod
    def generate_json(self, schema: dict[str, Any], system_prompt: str, user_prompt: str, timeout: int = 30) -> dict[str, Any]:
        raise NotImplementedError


class _BaseOpenAIAdapter(LLMClient):
    def __init__(self, model: str, failure_threshold: int = 5, cooldown_seconds: int = 60) -> None:
        self.model = model
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._open_until_epoch = 0.0

    def _check_circuit(self) -> None:
        now = time.time()
        if now < self._open_until_epoch:
            raise LLMCircuitOpenError("LLM circuit breaker is open. Try again later.")

    def _mark_success(self) -> None:
        self._failure_count = 0
        self._open_until_epoch = 0.0

    def _mark_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._open_until_epoch = time.time() + self.cooldown_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _request(self, system_prompt: str, user_prompt: str, timeout: int) -> dict[str, Any]:
        self._check_circuit()
        raw = self._chat_completion(system_prompt=system_prompt, user_prompt=user_prompt, timeout=timeout)
        return json.loads(raw)

    @abstractmethod
    def _chat_completion(self, system_prompt: str, user_prompt: str, timeout: int) -> str:
        raise NotImplementedError

    def generate_json(self, schema: dict[str, Any], system_prompt: str, user_prompt: str, timeout: int = 30) -> dict[str, Any]:
        schema_text = json.dumps(schema, ensure_ascii=True)
        wrapped_system_prompt = (
            f"{system_prompt}\n"
            "Return a single JSON object with no markdown.\n"
            f"Strict JSON Schema:\n{schema_text}"
        )
        try:
            payload = self._request(wrapped_system_prompt, user_prompt, timeout)
            validate_schema(payload, schema)
            self._mark_success()
            return payload
        except SchemaValidationError:
            self._mark_failure()
            raise
        except Exception:
            self._mark_failure()
            raise


class AzureOpenAIClient(_BaseOpenAIAdapter):
    def __init__(self, endpoint: str, api_key: str, deployment: str) -> None:
        super().__init__(model=deployment)
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-15-preview")

    def _chat_completion(self, system_prompt: str, user_prompt: str, timeout: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=timeout,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty content.")
        return content


class OpenAIClient(_BaseOpenAIAdapter):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(model=model)
        self.client = OpenAI(api_key=api_key)

    def _chat_completion(self, system_prompt: str, user_prompt: str, timeout: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=timeout,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty content.")
        return content


def create_llm_client_from_env() -> LLMClient:
    provider = os.getenv("AI_PROVIDER", "azure").strip().lower()

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    if provider == "azure":
        if azure_endpoint and azure_key and azure_deployment:
            return AzureOpenAIClient(endpoint=azure_endpoint, api_key=azure_key, deployment=azure_deployment)
        if openai_key:
            return OpenAIClient(api_key=openai_key, model=openai_model)
        raise RuntimeError("Azure configuration missing and OpenAI fallback is not configured.")

    if provider == "openai":
        if openai_key:
            return OpenAIClient(api_key=openai_key, model=openai_model)
        if azure_endpoint and azure_key and azure_deployment:
            return AzureOpenAIClient(endpoint=azure_endpoint, api_key=azure_key, deployment=azure_deployment)
        raise RuntimeError("OpenAI configuration missing and Azure fallback is not configured.")

    raise RuntimeError("Unsupported AI_PROVIDER. Use 'azure' or 'openai'.")
