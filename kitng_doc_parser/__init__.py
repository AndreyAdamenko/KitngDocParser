"""Универсальный конвейер парсинга документов в Markdown для MCP-серверов KITNG."""

from .models import DocumentRef, ParseResult, CacheEntry
from .concurrency import DocumentLockRegistry
from .pipeline import DocumentParser
from .cache.md_cache import MdCache
from .cache.chunk_cache import ChunkCache
from .ocr.openrouter import OpenRouterClient, OpenRouterError
from .extractors.docx import extract_docx_markdown
from .extractors.xlsx import extract_xlsx_markdown
from .extractors.text import extract_text_markdown

__version__ = "1.0.0"

__all__ = [
    "DocumentParser",
    "MdCache",
    "ChunkCache",
    "DocumentLockRegistry",
    "DocumentRef",
    "ParseResult",
    "CacheEntry",
    "OpenRouterClient",
    "OpenRouterError",
    "extract_docx_markdown",
    "extract_xlsx_markdown",
    "extract_text_markdown",
]
