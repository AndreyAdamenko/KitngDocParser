"""Утилиты для нарезки и локального извлечения текста из PDF."""

from __future__ import annotations

import io
from pypdf import PdfReader, PdfWriter


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
