"""Извлечение текста и таблиц DOCX в Markdown."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator, Union

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def _iter_block_items(document: Document) -> Iterator[Union[Paragraph, Table]]:
    """Итерирует параграфы и таблицы документа в исходном порядке."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def extract_docx_markdown(data_or_path: Union[bytes, str, Path]) -> str:
    """Преобразует содержимое DOCX/DOCM в форматированный Markdown."""
    if isinstance(data_or_path, (str, Path)):
        document = Document(str(data_or_path))
    else:
        document = Document(io.BytesIO(data_or_path))

    lines: list[str] = []

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue

            style_name = (block.style.name if block.style else "").lower()
            if "heading 1" in style_name:
                lines.append(f"# {text}\n")
            elif "heading 2" in style_name:
                lines.append(f"## {text}\n")
            elif "heading 3" in style_name:
                lines.append(f"### {text}\n")
            elif "list" in style_name:
                lines.append(f"- {text}")
            else:
                lines.append(text)
        elif isinstance(block, Table):
            lines.append("")
            rows = block.rows
            if not rows:
                continue

            # Первая строка как заголовок таблицы
            headers = [
                cell.text.strip().replace("\r", " ").replace("\n", " ").replace("|", "\\|")
                for cell in rows[0].cells
            ]
            if not any(headers):
                headers = [f"Колонка {i + 1}" for i in range(len(headers))]

            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for row in rows[1:]:
                cells = [
                    cell.text.strip().replace("\r", " ").replace("\n", " ").replace("|", "\\|")
                    for cell in row.cells
                ]
                if any(cells):
                    lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    return "\n".join(lines).strip()
