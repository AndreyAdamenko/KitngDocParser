"""Извлечение листов и таблиц Excel (XLSX, XLS) в Markdown."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import openpyxl


def extract_xlsx_markdown(
    data_or_path: Union[bytes, str, Path],
    max_rows_per_sheet: int = 500,
    max_cols: int = 40,
) -> str:
    """Преобразует книгу Excel в структурированный Markdown с таблицами по каждому листу."""
    if isinstance(data_or_path, (str, Path)):
        wb = openpyxl.load_workbook(str(data_or_path), read_only=True, data_only=True)
    else:
        wb = openpyxl.load_workbook(io.BytesIO(data_or_path), read_only=True, data_only=True)

    output: list[str] = []

    try:
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            output.append(f"## Лист: {sheet_name}\n")

            rows_data: list[list[str]] = []
            row_count = 0

            for row in sheet.iter_rows(values_only=True):
                row_count += 1
                if row_count > max_rows_per_sheet:
                    break

                # Ограничиваем количество колонок
                trimmed_row = list(row[:max_cols]) if row else []
                # Форматируем значения
                str_cells = []
                for val in trimmed_row:
                    if val is None:
                        str_cells.append("")
                    else:
                        cleaned = str(val).strip().replace("\r", " ").replace("\n", " ").replace("|", "\\|")
                        str_cells.append(cleaned)

                # Проверяем, есть ли хоть одна непустая ячейка в строке
                if any(str_cells):
                    # Обрезаем завершающие пустые ячейки
                    while str_cells and not str_cells[-1]:
                        str_cells.pop()
                    rows_data.append(str_cells)

            if not rows_data:
                output.append("*(Лист пуст)*\n")
                continue

            # Нормализуем ширину таблицы
            max_len = max(len(r) for r in rows_data)
            if max_len == 0:
                output.append("*(Лист пуст)*\n")
                continue

            normalized_rows = [r + [""] * (max_len - len(r)) for r in rows_data]

            # Первая непустая строка — заголовок
            headers = normalized_rows[0]
            # Если в заголовках все ячейки пустые, нумеруем
            display_headers = [
                h if h else f"Колонка {i + 1}" for i, h in enumerate(headers)
            ]

            output.append("| " + " | ".join(display_headers) + " |")
            output.append("| " + " | ".join(["---"] * max_len) + " |")

            for r in normalized_rows[1:]:
                output.append("| " + " | ".join(r) + " |")

            if row_count > max_rows_per_sheet:
                output.append(f"\n*(Показано первые {max_rows_per_sheet} строк листа)*\n")

            output.append("\n")
    finally:
        wb.close()

    return "\n".join(output).strip()
