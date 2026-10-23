"""Главный конвейер оркестрации парсеров документов."""

from __future__ import annotations

import logging
import hashlib
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .cache.chunk_cache import ChunkCache
from .extractors.docx import extract_docx_markdown
from .extractors.text import extract_text_markdown
from .extractors.xlsx import extract_xlsx_markdown
from .models import DocumentRef, ParseResult
from .ocr.openrouter import OpenRouterClient, OpenRouterError
from .pdf.worker import pdf_operation
from .quality import validate_markdown_quality, native_page_quality, QUALITY_POLICY
from .pdf.chunking import extract_pdf_pages, extract_pdf_raw_text, get_pdf_page_count

logger = logging.getLogger("kitng_doc_parser.pipeline")

DOCX_EXTS = {"docx", "docm"}
XLSX_EXTS = {"xlsx", "xls"}
PDF_EXTS = {"pdf"}
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}
TEXT_EXTS = {"txt", "csv", "log", "md", "json", "xml", "ini"}


def _ext(file_name: str) -> str:
    return Path(file_name).suffix.lower().lstrip(".")


def _escape(value: str) -> str:
    return value.replace('"', "'")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentParser:
    """Универсальный фасад парсинга инженерных и управленческих документов в Markdown."""

    def __init__(
        self,
        openrouter: Optional[OpenRouterClient] = None,
        chunk_threshold: int = 8,
        chunk_size: int = 6,
        parse_docx_via_llm: bool = False,
        parse_partial_fallback: bool = True,
    ) -> None:
        self._openrouter = openrouter
        self._chunk_threshold = chunk_threshold
        self._chunk_size = chunk_size
        self._parse_docx_via_llm = parse_docx_via_llm
        self._parse_partial_fallback = parse_partial_fallback

    async def parse(
        self,
        ref: DocumentRef,
        data: bytes,
        *,
        chunk_cache: Optional[ChunkCache] = None,
        force: bool = False,
        budget=None,
    ) -> ParseResult:
        """Определяет формат документа и преобразует его в форматированный Markdown."""
        ext = _ext(ref.file_name)

        if ext in DOCX_EXTS and not self._parse_docx_via_llm:
            return self._parse_docx_local(ref, data)
        if ext in XLSX_EXTS:
            return self._parse_xlsx_local(ref, data)
        if ext in TEXT_EXTS:
            return self._parse_text_local(ref, data)
        if ext in PDF_EXTS:
            return await self._parse_pdf(ref, data, chunk_cache, force, budget)
        if ext in IMAGE_EXTS:
            return await self._parse_image(ref, data)
        if ext in DOCX_EXTS:
            return await self._parse_via_llm(ref, data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # Фоллбэк для неизвестных форматов
        try:
            return self._parse_text_local(ref, data)
        except Exception:
            raise ValueError(f"Формат файла '.{ext}' не поддерживается для извлечения текста.")

    def _parse_docx_local(self, ref: DocumentRef, data: bytes) -> ParseResult:
        text = extract_docx_markdown(data)
        markdown = self._frontmatter(ref, pages=1, origin="local") + text
        return ParseResult(markdown=markdown, pages=1, origin="local", status="complete")

    def _parse_xlsx_local(self, ref: DocumentRef, data: bytes) -> ParseResult:
        text = extract_xlsx_markdown(data)
        markdown = self._frontmatter(ref, pages=1, origin="local") + text
        return ParseResult(markdown=markdown, pages=1, origin="local", status="complete")

    def _parse_text_local(self, ref: DocumentRef, data: bytes) -> ParseResult:
        text = extract_text_markdown(data, ref.file_name)
        markdown = self._frontmatter(ref, pages=1, origin="text") + text
        return ParseResult(markdown=markdown, pages=1, origin="text", status="complete")

    async def _parse_image(self, ref: DocumentRef, data: bytes) -> ParseResult:
        if not self._openrouter:
            raise ValueError("OpenRouter клиент не настроен для OCR изображений.")
        ext = _ext(ref.file_name)
        mime = "image/png" if ext == "png" else "image/jpeg"
        body = await self._openrouter.convert_to_markdown(data, mime)
        if not body or not body.strip():
            raise OpenRouterError("Модель вернула пустой ответ.", retryable=True)
        markdown = self._frontmatter(ref, pages=1, origin="ocr") + body
        return ParseResult(
            markdown=markdown,
            pages=1,
            origin="ocr",
            model=self._openrouter.model_name,
            status="complete",
        )

    async def _parse_via_llm(self, ref: DocumentRef, data: bytes, mime: str) -> ParseResult:
        if not self._openrouter:
            return self._parse_docx_local(ref, data)
        try:
            body = await self._openrouter.convert_to_markdown(data, mime)
        except OpenRouterError:
            return self._parse_docx_local(ref, data)
        if not body or not body.strip():
            return self._parse_docx_local(ref, data)
        markdown = self._frontmatter(ref, pages=1, origin="ocr") + body
        return ParseResult(
            markdown=markdown,
            pages=1,
            origin="ocr",
            model=self._openrouter.model_name,
            status="complete",
        )

    def processing_revision(self, data: bytes, cache_generation: str = "") -> str:
        policy = {"schema": 3, "generation": cache_generation, "parser": "1.2.0",
                  "quality": QUALITY_POLICY, "prompt": "ocr-v2-leaders", "ocr_page_size": 1,
                  "native_first": True, "model": self._openrouter.model_name if self._openrouter else None,
                  "route": getattr(self._openrouter, "_base_url", None)}
        return hashlib.sha256(hashlib.sha256(data).digest() + json.dumps(policy, sort_keys=True).encode()).hexdigest()

    async def _parse_pdf(self, ref, data, chunk_cache, force, budget=None) -> ParseResult:
        revision = self.processing_revision(data, ref.cache_generation or (_now() if force else ""))
        source_revision = hashlib.sha256(data).hexdigest()
        cache = chunk_cache.for_revision(revision) if chunk_cache is not None else None
        if force and cache: cache.clear(ref.scope, ref.item_id)
        manifest = cache.get_manifest(ref.scope, ref.item_id) if cache and not force else None
        if manifest and (manifest.get("quality_policy") != QUALITY_POLICY or manifest.get("source_revision") != source_revision):
            manifest = None
        memory = {}
        def get(index):
            return cache.get_chunk(ref.scope, ref.item_id, index) if cache else memory.get(index)
        def save(index, text):
            if budget: budget.check()
            if cache: cache.save_chunk(ref.scope, ref.item_id, index, text)
            else: memory[index] = text
        def persist():
            if budget: budget.check_cancelled()
            if cache: cache.save_manifest(ref.scope, ref.item_id, manifest)

        inspect = not manifest or not manifest.get("analysis_complete")
        if manifest:
            total_pages = manifest.get("total_pages")
            chunks = manifest.get("chunks", [])
            if not isinstance(total_pages, int) or total_pages <= 0 or len(chunks) != total_pages or any(c.get("index") != i for i, c in enumerate(chunks)):
                manifest, inspect = None, True
            else:
                for meta in chunks:
                    if meta.get("status") != "done": continue
                    text = get(meta["index"])
                    checked = validate_markdown_quality(text, page_count=1)
                    if not checked.valid or meta.get("origin") not in {"native_pdf", "ocr"}:
                        if meta.get("origin") == "native_pdf": inspect = True
                        meta.update(status="pending", invalidated_reason=checked.reason_code or "unknown_origin")

        if inspect:
            info = await pdf_operation("inspect", data, budget)
            total_pages = info["total_pages"]
            if total_pages == 0:
                return ParseResult(self._frontmatter(ref, 0, "partial", "partial"), 0, "partial",
                    status="partial", completed_chunks=0, total_chunks=0, error="empty_pdf",
                    reason_code="empty_content", source_revision=source_revision,
                    processing_revision=revision, quality_policy=QUALITY_POLICY)
            if manifest is None:
                manifest = {"scope": ref.scope, "item_id": ref.item_id, "total_pages": total_pages,
                    "total_chunks": total_pages, "source_revision": source_revision,
                    "quality_policy": QUALITY_POLICY, "analysis_complete": False, "status": "in_progress",
                    "chunks": [{"index": i, "pages": f"{i+1}-{i+1}", "status": "pending"} for i in range(total_pages)]}
            # Complete all cheap native pages in one slice; max_new_chunks limits
            # paid OCR calls, not local page checkpoints.
            for index, page_info in enumerate(info["pages"]):
                if budget: budget.check()
                meta = manifest["chunks"][index]
                if meta.get("status") == "done": continue
                checked = native_page_quality(page_info)
                meta["native_quality"] = checked.reason_code or "usable"
                if checked.valid:
                    text = page_info["text"]
                    save(index, text)
                    meta.update(status="done", origin="native_pdf", chars=len(text), parsed_at=_now(), quality_policy=QUALITY_POLICY)
            manifest["analysis_complete"] = True
            persist()

        chunks = manifest["chunks"]
        failed_error = reason_code = None
        failed_index = retry_after = None
        retryable = False
        new_ocr_pages = 0
        for index, meta in enumerate(chunks):
            if meta.get("status") == "done": continue
            if budget and new_ocr_pages >= budget.max_new_chunks: break
            if not self._openrouter or not getattr(self._openrouter, "_api_key", None):
                failed_error, reason_code, failed_index = "OCR is required for remaining pages but is not configured", "ocr_not_configured", index
                break
            try:
                if budget: budget.check()
                chunk_data = await pdf_operation("chunk", data, budget, index + 1, index + 1)
                prompt = f"Распознай страницу {index+1} из {total_pages} документа '{ref.file_name}' и переведи её в точный Markdown."
                text = await self._openrouter.convert_to_markdown(chunk_data, "application/pdf", prompt,
                    **({"budget": budget} if budget else {}))
                if budget: budget.check()
                new_ocr_pages += 1
                checked = validate_markdown_quality(text, page_count=1)
                if not checked.valid:
                    raise OpenRouterError("OCR quality rejected: " + checked.reason_code, reason_code="invalid_content")
            except TimeoutError:
                failed_error, reason_code, failed_index, retryable = "slice_deadline", "slice_deadline", index, True
                break
            except OpenRouterError as ex:
                failed_error, failed_index, retryable, retry_after = ex.message, index, ex.retryable, ex.retry_after
                reason_code = ex.reason_code or ("provider_rate_limited" if ex.status == 429 else "provider_unavailable")
                meta.update(status="failed", reason_code=reason_code, retryable=retryable, parsed_at=_now())
                logger.warning("OCR page %s rejected: %s", index + 1, reason_code)
                break
            save(index, text)
            meta.update(status="done", origin="ocr", model=self._openrouter.model_name,
                        chars=len(text), parsed_at=_now(), quality_policy=QUALITY_POLICY)
            persist()

        # Revalidate every reused checkpoint before assembly, even in a current
        # revision. No fallback/native text is silently substituted for failures.
        parts, origins = [], {}
        for index, meta in enumerate(chunks):
            text = get(index) if meta.get("status") == "done" else None
            checked = validate_markdown_quality(text, page_count=1)
            if text is not None and checked.valid and meta.get("origin") in {"native_pdf", "ocr"}:
                origins[index + 1] = meta["origin"]
                parts.append(f"<!-- Pages {index+1} - {index+1}; origin={meta['origin']} -->\n\n{text}")
            else:
                if meta.get("status") == "done": meta.update(status="pending", invalidated_reason=checked.reason_code)
                origins[index + 1] = "missing"
                parts.append(f"<!-- Pages {index+1} - {index+1}: не распознано -->")
        native_pages = sum(value == "native_pdf" for value in origins.values())
        ocr_pages = sum(value == "ocr" for value in origins.values())
        done = native_pages + ocr_pages
        complete = done == total_pages and failed_error is None
        status = "complete" if complete else "partial"
        origin = ("native_pdf" if native_pages == total_pages else "ocr" if ocr_pages == total_pages else "mixed_pdf") if complete else "partial"
        manifest["status"] = status
        persist()
        markdown = self._frontmatter(ref, total_pages, origin, status, done, total_pages) + "\n\n".join(parts)
        return ParseResult(markdown=markdown, pages=total_pages, origin=origin,
            model=self._openrouter.model_name if self._openrouter and ocr_pages else None,
            status=status, completed_chunks=done, total_chunks=total_pages, failed_chunk=failed_index,
            retryable=retryable, error=failed_error, retry_after=retry_after,
            source_revision=source_revision, processing_revision=revision, reason_code=reason_code,
            native_pages=native_pages, ocr_pages=ocr_pages, page_origins=origins, quality_policy=QUALITY_POLICY)


    @staticmethod
    def _frontmatter(
        ref: DocumentRef,
        pages: int,
        origin: str,
        status: str = "complete",
        completed_chunks: Optional[int] = None,
        total_chunks: Optional[int] = None,
    ) -> str:
        lines = [
            "---",
            f'title: "{_escape(ref.file_name)}"',
            f'source_file: "{_escape(ref.item_id)}"',
            f'scope: "{_escape(ref.scope)}"',
            f"pages: {pages}",
            f'origin: "{origin}"',
            f'status: "{status}"',
        ]
        if ref.mtime:
            lines.append(f"mtime: {ref.mtime}")
        if ref.size_bytes:
            lines.append(f"size_bytes: {ref.size_bytes}")
        if completed_chunks is not None and total_chunks is not None:
            lines.append(f"completed_chunks: {completed_chunks}")
            lines.append(f"total_chunks: {total_chunks}")
        lines.append("---")
        return "\n".join(lines) + "\n\n"
