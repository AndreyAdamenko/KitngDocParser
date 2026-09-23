"""Модели данных для конвейера парсинга."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DocumentRef:
    """Идентификатор и метаданные документа для парсинга."""

    scope: str
    """Область видимости / проект (например, '5837' или 'contract_102')."""

    item_id: str
    """Идентификатор файла внутри scope (относительный путь, имя файла или ID)."""

    file_name: str
    """Оригинальное имя файла с расширением."""

    file_path: Optional[str] = None
    """Абсолютный или относительный физический путь на диске (если применимо)."""

    mtime: Optional[float] = None
    """Временная метка последней модификации файла (Unix epoch)."""

    size_bytes: Optional[int] = None
    """Размер файла в байтах."""

    content_type: Optional[str] = None
    """MIME-тип (если известен)."""


@dataclass
class ParseResult:
    """Результат парсинга документа в Markdown."""

    markdown: str
    pages: int
    origin: str  # "local", "ocr", "fallback", "text"
    model: Optional[str] = None
    status: str = "complete"  # "complete", "partial", "fallback", "failed"
    completed_chunks: Optional[int] = None
    total_chunks: Optional[int] = None
    failed_chunk: Optional[int] = None
    retryable: bool = False
    error: Optional[str] = None


@dataclass
class CacheEntry:
    """Запись из кэша Markdown."""

    markdown: str
    meta: Dict[str, Any] = field(default_factory=dict)
