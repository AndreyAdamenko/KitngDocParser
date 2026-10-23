"""Асинхронный клиент OpenRouter / Gemini API для OCR документов."""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

from ..admission import shared_admission, ProviderCooldown
from .prompts import OCR_SYSTEM_PROMPT
from ..quality import validate_markdown_quality, MAX_PAGE_CHARS, MAX_OUTPUT_TOKENS, MAX_RESPONSE_BYTES

logger = logging.getLogger("kitng_doc_parser.ocr")

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 30.0
DEFAULT_429_COOLDOWN_SECONDS = 60.0


class OpenRouterError(Exception):
    """Ошибка вызова OCR модели."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: Optional[int] = None,
        retry_after: Optional[float] = None,
        reason_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after
        self.reason_code = reason_code


def _clean_markdown(text: str) -> str:
    trimmed = text.strip()
    for prefix in ("```markdown", "```json", "```"):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):]
            break
    if trimmed.endswith("```"):
        trimmed = trimmed[:-3]
    return trimmed.strip()


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


class OpenRouterClient:
    """Клиент для распознавания документов через OpenRouter Vision / Gemini."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "google/gemini-2.5-flash-lite",
        base_url: str = "https://openrouter.ai/api/v1/",
        timeout_seconds: float = 180.0,
        max_attempts: int = 3,
        retry_base_seconds: float = 2.0,
        max_in_flight: int = 1,
        admission=None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        max_output_chars: int = MAX_PAGE_CHARS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        self._admission = admission
        self._max_in_flight = max_in_flight
        self._max_output_tokens = max(1, int(max_output_tokens))
        self._max_output_chars = max(1, int(max_output_chars))
        self._max_response_bytes = max(1, int(max_response_bytes))
        self._base_url = base_url
        self._api_key = api_key
        self._model_name = model_name
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://kitng.kz",
                "X-Title": "KITNG MCP Document Parser",
            },
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int, base: float) -> float:
        if base <= 0:
            return 0.0
        delay = base * (2 ** (attempt - 1)) + random.uniform(0, base)
        return min(delay, MAX_BACKOFF_SECONDS)

    async def _bounded_post(self, payload, timeout):
        async with self._client.stream("POST", "chat/completions", json=payload, timeout=timeout) as response:
            # aiter_bytes() returns decoded bytes. Reusing Content-Encoding on
            # the materialized response would attempt gzip/br decoding twice.
            headers = {key: value for key, value in response.headers.items()
                       if key.lower() not in {"content-encoding", "content-length"}}
            # Error bodies may contain documents; their contents are unnecessary.
            if response.status_code != 200:
                return httpx.Response(response.status_code, headers=headers)
            body = bytearray()
            async for block in response.aiter_bytes():
                if len(body) + len(block) > self._max_response_bytes:
                    raise OpenRouterError("OCR quality rejected: response_size_limit", reason_code="invalid_content")
                body.extend(block)
            return httpx.Response(response.status_code, headers=headers, content=bytes(body))

    async def _wait_retry(self, delay, budget, reason_code):
        if delay <= 0: return
        if budget and delay >= budget.remaining():
            raise OpenRouterError("Provider retry deferred beyond slice budget", retryable=True,
                                  retry_after=delay, reason_code=reason_code)
        if budget:
            await asyncio.wait_for(asyncio.sleep(delay), budget.remaining())
        else:
            await asyncio.sleep(delay)

    async def convert_to_markdown(
        self,
        data: bytes,
        mime_type: str,
        prompt: Optional[str] = None,
        *, budget=None,
    ) -> str:
        if not self._api_key:
            raise OpenRouterError("OPENROUTER_API_KEY не задан.", retryable=False)

        data_url = f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
        if mime_type == "application/pdf":
            file_object = {"type": "file", "file": {"file_data": data_url}}
        else:
            file_object = {"type": "image_url", "image_url": {"url": data_url}}

        user_text = prompt or (
            "Распознай этот документ и переведи его в точный Markdown формат "
            "согласно системным инструкциям. Сохрани все таблицы, пункты и реквизиты."
        )

        payload = {
            "model": self._model_name,
            "temperature": 0.1,
            "max_tokens": self._max_output_tokens,
            "messages": [
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        file_object,
                    ],
                },
            ],
        }

        admission = self._admission or shared_admission(self._api_key, self._model_name, self._base_url, self._max_in_flight)
        attempts = max(1, budget.max_attempts if budget else self._max_attempts)
        base = max(0.0, float(self._retry_base_seconds))
        last_error = ""
        last_status: Optional[int] = None
        last_retry_after: Optional[float] = None

        for attempt in range(1, attempts + 1):
            try:
                async with admission.slot(budget) as started_epoch:
                    timeout = min(self._client.timeout.read or 180.0, budget.remaining()) if budget else self._client.timeout
                    request = self._bounded_post(payload, timeout)
                    response = await asyncio.wait_for(request, budget.remaining()) if budget else await request
                    if response.status_code == 200:
                        admission.succeeded(started_epoch)
                    elif response.status_code in RETRYABLE_STATUS:
                        requested_delay = _parse_retry_after(response.headers.get("retry-after"))
                        if response.status_code == 429 and requested_delay is None:
                            requested_delay = DEFAULT_429_COOLDOWN_SECONDS
                        last_retry_after = admission.failed(requested_delay, base)
            except ProviderCooldown as ex:
                raise OpenRouterError("Provider cooldown", retryable=True, status=429, retry_after=ex.retry_after, reason_code="provider_rate_limited") from ex
            except (httpx.HTTPError, TimeoutError) as ex:
                last_retry_after = admission.failed(None, base)
                last_error = type(ex).__name__
                logger.warning("Сетевая ошибка OpenRouter (попытка %s/%s): %s", attempt, attempts, ex)
                if attempt < attempts:
                    delay = last_retry_after if last_retry_after is not None else self._backoff(attempt, base)
                    await self._wait_retry(delay, budget, "provider_timeout" if isinstance(ex, (TimeoutError, httpx.TimeoutException)) else "provider_unavailable")
                    continue
                code = "provider_timeout" if isinstance(ex, (TimeoutError, httpx.TimeoutException)) else "provider_unavailable"
                raise OpenRouterError(f"OpenRouter transport unavailable after {attempts} attempts: {last_error}", retryable=True, retry_after=last_retry_after, reason_code=code) from ex

            if response.status_code == 200:
                try:
                    body = response.json()
                    choice = body["choices"][0]
                    content = choice["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as ex:
                    raise OpenRouterError("Invalid OpenRouter response schema", reason_code="invalid_content") from ex
                checked = validate_markdown_quality(content, finish_reason=choice.get("finish_reason"),
                    max_chars=self._max_output_chars, require_finish_reason=True)
                tokens = body.get("usage", {}).get("completion_tokens") if isinstance(body.get("usage", {}), dict) else None
                if not checked.valid:
                    raise OpenRouterError("OCR quality rejected: " + checked.reason_code, reason_code="invalid_content")
                if isinstance(tokens, (int, float)) and tokens > self._max_output_tokens:
                    raise OpenRouterError("OCR quality rejected: output_token_limit", reason_code="invalid_content")
                cleaned = _clean_markdown(content)
                checked = validate_markdown_quality(cleaned, max_chars=self._max_output_chars)
                if not checked.valid:
                    raise OpenRouterError("OCR quality rejected: " + checked.reason_code, reason_code="invalid_content")
                return cleaned

            last_status = response.status_code
            last_retry_after = max(admission.retry_after, _parse_retry_after(response.headers.get("retry-after")) or 0.0)
            last_error = str(response.status_code)  # provider bodies can contain document data; never log them
            retryable = response.status_code in RETRYABLE_STATUS
            logger.warning("OpenRouter вернул %s (попытка %s/%s, retryable=%s)", response.status_code, attempt, attempts, retryable)
            if retryable and attempt < attempts:
                delay = last_retry_after if last_retry_after is not None else self._backoff(attempt, base)
                await self._wait_retry(delay, budget, "provider_rate_limited" if last_status == 429 else "provider_unavailable")
                continue
            raise OpenRouterError(f"OpenRouter error {last_error}", retryable=retryable, status=last_status, retry_after=last_retry_after,
                                  reason_code="provider_rate_limited" if last_status == 429 else "provider_unavailable")

        raise OpenRouterError(f"OpenRouter недоступен: {last_error}", retryable=True, status=last_status, retry_after=last_retry_after)
