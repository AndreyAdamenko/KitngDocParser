"""Универсальный конвейер парсинга документов в Markdown для MCP-серверов KITNG."""

from .budget import ParseBudget
from .admission import ProviderAdmission
from .quality import QualityCheck, validate_markdown_quality, QUALITY_POLICY
from .models import DocumentRef, ParseResult, CacheEntry
from .concurrency import DocumentLockRegistry
from .pipeline import DocumentParser
from .cache.md_cache import MdCache
from .cache.chunk_cache import ChunkCache
from .ocr.openrouter import OpenRouterClient, OpenRouterError
from .extractors.docx import extract_docx_markdown
from .extractors.xlsx import extract_xlsx_markdown
from .extractors.text import extract_text_markdown

__version__ = "1.2.0"

__all__ = [
    "QualityCheck", "validate_markdown_quality", "QUALITY_POLICY",
    "ParseBudget",
    "ProviderAdmission",
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
