"""Утилиты для нарезки и локального извлечения текста из PDF."""

from __future__ import annotations

import io
from pypdf import PdfReader, PdfWriter
from ..quality import MAX_PAGE_CHARS, MAX_DOCUMENT_CHARS


def get_pdf_page_count(data: bytes) -> int:
    """Возвращает количество страниц в PDF-документе."""
    reader = PdfReader(io.BytesIO(data))
    return len(reader.pages)


def extract_pdf_pages(data: bytes, start_page: int, end_page: int) -> bytes:
    """Вырезает диапазон страниц [start_page, end_page] (1-based, включительно)."""
    reader = PdfReader(io.BytesIO(data))
    writer = PdfWriter()
    for idx in range(start_page - 1, min(end_page, len(reader.pages))):
        writer.add_page(reader.pages[idx])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def extract_pdf_raw_text(data: bytes) -> str:
    """Локальное извлечение текста из страниц PDF через pypdf (фоллбэк)."""
    reader = PdfReader(io.BytesIO(data))
    pages_text: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages_text.append(f"<!-- Стр. {idx} -->\n\n{txt}")
    return "\n\n---\n\n".join(pages_text) if pages_text else "*(Текстовый слой PDF пуст или файл отсканирован)*"


def inspect_pdf_pages(data: bytes) -> dict:
    """One reader/one bounded child for the whole native-text pass.

    Text and diagnostics are capped before IPC. Image placement uses the PDF
    current transformation matrix, rather than guessing from pixel dimensions.
    Unknown image/form analysis fails closed to per-page OCR.
    """
    reader = PdfReader(io.BytesIO(data))
    result, remaining = [], MAX_DOCUMENT_CHARS
    for page in reader.pages:
        images = {}
        uncertain = False
        def collect(resources, depth=0):
            nonlocal uncertain
            if depth > 12:
                uncertain = True
                return
            try:
                resources = resources.get_object() if hasattr(resources, "get_object") else resources
                objects = resources.get("/XObject", {}) if resources else {}
                objects = objects.get_object() if hasattr(objects, "get_object") else objects
                for name, indirect in objects.items():
                    obj = indirect.get_object()
                    if obj.get("/Subtype") == "/Image": images[str(name)] = True
                    elif obj.get("/Subtype") == "/Form": collect(obj.get("/Resources", {}), depth + 1)
            except Exception:
                uncertain = True
        collect(page.get("/Resources", {}))
        area = max(1.0, float(page.mediabox.width) * float(page.mediabox.height))
        coverage = 0.0
        seen_images = set()
        def operand(operator, operands, cm, tm):
            nonlocal coverage, uncertain
            if operator != b"Do" or not operands: return
            name = str(operands[0])
            if name in images:
                seen_images.add(name)
                try:
                    coverage += abs(float(cm[0]) * float(cm[3]) - float(cm[1]) * float(cm[2])) / area
                except (ValueError, TypeError, IndexError):
                    uncertain = True
        try:
            raw = (page.extract_text(visitor_operand_before=operand) or "").strip() if remaining > 0 else ""
        except Exception:
            raw, uncertain = "", True
        limit = min(MAX_PAGE_CHARS, remaining)
        truncated = len(raw) > limit or remaining <= 0
        text = raw[:limit]
        remaining -= len(text)
        if images and not seen_images: uncertain = True
        result.append({"text": text, "text_truncated": truncated,
                       "image_coverage": min(1.0, coverage),
                       "image_analysis_uncertain": uncertain})
    return {"total_pages": len(reader.pages), "pages": result}
