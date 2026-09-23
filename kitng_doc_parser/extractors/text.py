"""Извлечение простых текстовых документов (TXT, CSV, LOG)."""

import csv
import io


def extract_text_markdown(data: bytes, file_name: str) -> str:
    """Преобразует текстовый файл в Markdown с авто-определением кодировки (UTF-8, CP1251)."""
    text = ""
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if not text:
        text = data.decode("utf-8", errors="replace")

    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        try:
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if rows:
                max_cols = max(len(r) for r in rows)
                normalized = [r + [""] * (max_cols - len(r)) for r in rows]
                headers = [h.replace("|", "\\|") for h in normalized[0]]
                lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * max_cols) + " |",
                ]
                for r in normalized[1:]:
                    cells = [c.replace("|", "\\|") for c in r]
                    lines.append("| " + " | ".join(cells) + " |")
                return "\n".join(lines)
        except Exception:
            pass

    return f"```\n{text.strip()}\n```"
