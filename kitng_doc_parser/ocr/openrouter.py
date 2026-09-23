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

from .prompts import OCR_SYSTEM_PROMPT

logger = logging.getLogger("kitng_doc_parser.ocr")

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 30.0


class OpenRouterError(Exception):
    """Ошибка вызова OCR модели."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: Optional[int] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


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
    ) -> None:
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

    async def convert_to_markdown(
        self,
        data: bytes,
        mime_type: str,
        prompt: Optional[str] = None,
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

        attempts = max(1, self._max_attempts)
        base = max(0.0, float(self._retry_base_seconds))
        last_error = ""
        last_status: Optional[int] = None
        last_retry_after: Optional[float] = None

        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.post("chat/completions", json=payload)
            except httpx.HTTPError as ex:
                last_error = str(ex)
                logger.warning("Сетевая ошибка OpenRouter (попытка %s/%s): %s", attempt, attempts, ex)
                if attempt < attempts:
                    delay = self._backoff(attempt, base)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                raise OpenRouterError(f"OpenRouter недоступен после {attempts} попыток: {last_error}", retryable=True) from ex

            if response.status_code == 200:
                try:
                    body = response.json()
                    content = body["choices"][0]["message"]["content"] or ""
                except (ValueError, KeyError, IndexError, TypeError) as ex:
                    raise OpenRouterError(f"Некорректный ответ OpenRouter: {ex}") from ex
                return _clean_markdown(content)

            last_status = response.status_code
            last_retry_after = _parse_retry_after(response.headers.get("retry-after"))
            last_error = f"{response.status_code}: {response.text[:300]}"
            retryable = response.status_code in RETRYABLE_STATUS
            logger.warning("OpenRouter вернул %s (попытка %s/%s, retryable=%s)", response.status_code, attempt, attempts, retryable)
            if retryable and attempt < attempts:
                delay = last_retry_after if last_retry_after is not None else self._backoff(attempt, base)
                if delay > 0:
                    await asyncio.sleep(delay)
                continue
            raise OpenRouterError(f"OpenRouter error {last_error}", retryable=retryable, status=last_status, retry_after=last_retry_after)

        raise OpenRouterError(f"OpenRouter недоступен: {last_error}", retryable=True, status=last_status, retry_after=last_retry_after)
