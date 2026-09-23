"""Главный конвейер оркестрации парсеров документов."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .cache.chunk_cache import ChunkCache
from .extractors.docx import extract_docx_markdown
from .extractors.text import extract_text_markdown
from .extractors.xlsx import extract_xlsx_markdown
from .models import DocumentRef, ParseResult
from .ocr.openrouter import OpenRouterClient, OpenRouterError
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
            return await self._parse_pdf(ref, data, chunk_cache, force)
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

    async def _parse_pdf(
        self,
        ref: DocumentRef,
        data: bytes,
        chunk_cache: Optional[ChunkCache],
        force: bool,
    ) -> ParseResult:
        try:
            total_pages = get_pdf_page_count(data)
        except Exception as ex:
            raise ValueError(f"Не удалось прочитать PDF: {ex}") from ex

        if total_pages == 0:
            return ParseResult(
                markdown=self._frontmatter(ref, 0, "fallback") + "*(Пустой PDF)*",
                pages=0,
                origin="fallback",
                status="complete",
            )

        # Если нет OpenRouter или API-ключа, сразу используем быстрый локальный pypdf
        if not self._openrouter or not self._openrouter._api_key:
            raw = extract_pdf_raw_text(data)
            markdown = self._frontmatter(ref, total_pages, "local_pdf") + raw
            return ParseResult(
                markdown=markdown,
                pages=total_pages,
                origin="local_pdf",
                status="complete",
            )

        if chunk_cache is None:
            return await self._parse_pdf_simple(ref, data, total_pages)

        # Чанковый возобновляемый парсинг через OCR
        plan: List[Tuple[int, int]] = []
        for start in range(1, total_pages + 1, self._chunk_size):
            end = min(start + self._chunk_size - 1, total_pages)
            plan.append((start, end))

        total_chunks = len(plan)
        manifest = None if force else chunk_cache.get_manifest(ref.scope, ref.item_id)
        if manifest is None:
            manifest = {
                "scope": ref.scope,
                "item_id": ref.item_id,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "status": "in_progress",
                "chunks": [
                    {"index": i, "pages": f"{s}-{e}", "status": "pending"}
                    for i, (s, e) in enumerate(plan)
                ],
            }
            chunk_cache.save_manifest(ref.scope, ref.item_id, manifest)

        chunks_meta = manifest.setdefault("chunks", [])
        by_index = {c.get("index"): c for c in chunks_meta}

        failed_error: Optional[str] = None
        failed_index: Optional[int] = None
        retryable = False

        for index, (start, end) in enumerate(plan):
            meta = by_index.setdefault(index, {"index": index, "pages": f"{start}-{end}", "status": "pending"})
            if meta.get("status") == "done" and chunk_cache.get_chunk(ref.scope, ref.item_id, index) is not None:
                continue

            try:
                chunk_data = extract_pdf_pages(data, start, end)
                prompt = (
                    f"Распознай страницы {start}–{end} из {total_pages} документа "
                    f"'{ref.file_name}' и переведи их в точный Markdown."
                )
                chunk_md = await self._openrouter.convert_to_markdown(chunk_data, "application/pdf", prompt)
                if not chunk_md or not chunk_md.strip():
                    raise OpenRouterError("Модель вернула пустой ответ.", retryable=True)
            except OpenRouterError as ex:
                logger.warning("OCR чанка %s (стр. %s–%s) для %s не удался: %s", index, start, end, ref.file_name, ex)
                meta.update({"status": "failed", "error": ex.message, "retryable": ex.retryable, "parsed_at": _now()})
                failed_error = ex.message
                failed_index = index
                retryable = ex.retryable
                chunk_cache.save_manifest(ref.scope, ref.item_id, manifest)
                break

            chunk_cache.save_chunk(ref.scope, ref.item_id, index, chunk_md)
            meta.update({
                "status": "done",
                "origin": "ocr",
                "model": self._openrouter.model_name,
                "chars": len(chunk_md),
                "parsed_at": _now(),
            })
            chunk_cache.save_manifest(ref.scope, ref.item_id, manifest)

        done = sum(1 for c in chunks_meta if c.get("status") == "done")
        complete = failed_error is None and done == total_chunks
        manifest["status"] = "complete" if complete else "partial"
        chunk_cache.save_manifest(ref.scope, ref.item_id, manifest)

        parts: List[str] = []
        for index, (start, end) in enumerate(plan):
            meta = by_index.get(index) or {}
            if meta.get("status") == "done":
                cached = chunk_cache.get_chunk(ref.scope, ref.item_id, index)
                if cached is not None:
                    parts.append(f"<!-- Pages {start} - {end} -->\n\n{cached}")
                    continue
            if self._parse_partial_fallback:
                raw = extract_pdf_raw_text(extract_pdf_pages(data, start, end))
                parts.append(f"<!-- Pages {start} - {end} (pypdf fallback) -->\n\n{raw}")
            else:
                parts.append(f"<!-- Pages {start} - {end}: не распознано -->")

        status = "complete" if complete else "partial"
        origin = "ocr" if complete else "partial"
        markdown = self._frontmatter(ref, total_pages, origin, status, done, total_chunks) + "\n\n".join(parts)
        return ParseResult(
            markdown=markdown,
            pages=total_pages,
            origin=origin,
            model=self._openrouter.model_name if complete else None,
            status=status,
            completed_chunks=done,
            total_chunks=total_chunks,
            failed_chunk=failed_index,
            retryable=retryable if not complete else False,
            error=failed_error,
        )

    async def _parse_pdf_simple(self, ref: DocumentRef, data: bytes, total_pages: int) -> ParseResult:
        if not self._openrouter:
            raw = extract_pdf_raw_text(data)
            return ParseResult(markdown=self._frontmatter(ref, total_pages, "fallback") + raw, pages=total_pages, origin="fallback")

        try:
            if total_pages <= self._chunk_threshold:
                prompt = f"Распознай все {total_pages} стр. документа '{ref.file_name}' и переведи в Markdown."
                body = await self._openrouter.convert_to_markdown(data, "application/pdf", prompt)
                return ParseResult(markdown=self._frontmatter(ref, total_pages, "ocr") + body, pages=total_pages, origin="ocr", model=self._openrouter.model_name)

            parts = []
            for start in range(1, total_pages + 1, self._chunk_size):
                end = min(start + self._chunk_size - 1, total_pages)
                chunk = extract_pdf_pages(data, start, end)
                prompt = f"Распознай страницы {start}–{end} из {total_pages} документа '{ref.file_name}' и переведи в Markdown."
                chunk_md = await self._openrouter.convert_to_markdown(chunk, "application/pdf", prompt)
                parts.append(f"<!-- Pages {start} - {end} -->\n\n{chunk_md}")

            return ParseResult(markdown=self._frontmatter(ref, total_pages, "ocr") + "\n\n---\n\n".join(parts), pages=total_pages, origin="ocr", model=self._openrouter.model_name)
        except OpenRouterError as ex:
            raw = extract_pdf_raw_text(data)
            return ParseResult(markdown=self._frontmatter(ref, total_pages, "fallback") + raw, pages=total_pages, origin="fallback", status="fallback", retryable=ex.retryable, error=ex.message)

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
