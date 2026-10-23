"""Deterministic resource/degeneration guards, not a semantic accuracy score.

Reject suspect output without rewriting source numbers, table cells or leaders.
The limits are operational safety ceilings; passing them does not certify OCR.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

QUALITY_POLICY = "page-native-ocr-quality-v2"
MAX_PAGE_CHARS = 65_536
MAX_DOCUMENT_CHARS = 16 * 1024 * 1024
MAX_OUTPUT_TOKENS = 16_384
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_REPEAT_RUN = 512


@dataclass(frozen=True)
class QualityCheck:
    valid: bool
    reason_code: Optional[str]
    metrics: dict[str, Any]


_RUN = re.compile(r"([^\s])\1{511,}")
_DOT_RUN = re.compile(r"(?:[.\u2026\u00b7\u2024\uff0e][ \t]*){512,}")
_EMPTY_CELLS = re.compile(r"(?:\|[ \t]*){128,}")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_EMPTY_ROW = re.compile(r"^\s*\|[\s|:\-]*\|\s*$")


def validate_markdown_quality(
    text: str,
    *,
    finish_reason: Optional[str] = None,
    page_count: Optional[int] = None,
    max_chars: Optional[int] = None,
    require_finish_reason: bool = False,
) -> QualityCheck:
    """Validate OCR or a whole cached Markdown before slicing it.

    With no page count the document ceiling is used, so legacy assembled files
    can be checked too. Metadata/comments are not evidence of recognized text.
    A partial document with real text and missing-page comments is permitted;
    comments/placeholders alone are empty. No content is corrected or truncated.
    """
    metrics: dict[str, Any] = {"chars": len(text) if isinstance(text, str) else 0}
    def fail(reason): return QualityCheck(False, reason, metrics)
    if not isinstance(text, str): return fail("invalid_content_type")
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return fail("truncated_output")
    if finish_reason in {"content_filter", "error"}:
        return fail("filtered_output")
    if (require_finish_reason and not finish_reason) or (finish_reason and finish_reason != "stop"):
        return fail("invalid_finish_reason")
    ceiling = max_chars if max_chars is not None else min(MAX_DOCUMENT_CHARS, MAX_PAGE_CHARS * max(1, page_count)) if page_count is not None else MAX_DOCUMENT_CHARS
    metrics["max_chars"] = ceiling
    if len(text) > ceiling: return fail("output_size_limit")
    body = text.strip()
    if body.startswith("---\n") or body.startswith("---\r\n"):
        lines = body.splitlines()
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                body = "\n".join(lines[index + 1:])
                break
    body = _COMMENT.sub("", body).strip()
    if body in {"*(Пустой PDF)*", "*(Текстовый слой PDF пуст или файл отсканирован)*"}:
        return fail("empty_content")
    if not body or not any(char.isalnum() for char in body):
        return fail("empty_content")
    run = _RUN.search(body)
    if run:
        metrics["repeated_run_at_least"] = len(run.group(0))
        return fail("repeated_character_run")
    if _DOT_RUN.search(body): return fail("repeated_dot_leaders")
    if _EMPTY_CELLS.search(body): return fail("empty_table_repetition")
    empty_streak = max_empty = empty_rows = nonblank = 0
    for line in body.splitlines():
        if not line.strip(): continue
        nonblank += 1
        if _EMPTY_ROW.fullmatch(line):
            empty_rows += 1
            empty_streak += 1
            max_empty = max(max_empty, empty_streak)
        else:
            empty_streak = 0
    metrics.update(empty_table_rows=empty_rows, max_empty_table_streak=max_empty)
    if max_empty >= 128 or (max_empty >= 32 and empty_rows > nonblank * 0.8):
        return fail("empty_table_repetition")
    return QualityCheck(True, None, metrics)


def native_page_quality(info: dict) -> QualityCheck:
    text = info.get("text", "")
    checked = validate_markdown_quality(text, page_count=1)
    metrics = dict(checked.metrics)
    if not checked.valid: return checked
    if info.get("text_truncated"):
        return QualityCheck(False, "native_text_limit", metrics)
    visible = [c for c in text if not c.isspace()]
    alnum = sum(c.isalnum() for c in visible)
    bad = sum(c == "\ufffd" or (not c.isprintable()) or (0xE000 <= ord(c) <= 0xF8FF) for c in visible)
    metrics.update(alnum_chars=alnum, visible_chars=len(visible), suspicious_chars=bad,
                   image_coverage=info.get("image_coverage", 0.0))
    reason = None
    if len(visible) < 40 or alnum < 20: reason = "native_text_sparse"
    elif bad > max(0, len(visible) * 0.002): reason = "native_text_encoding"
    elif info.get("image_analysis_uncertain") or info.get("image_coverage", 0.0) >= 0.2:
        reason = "native_page_has_raster_content"
    return QualityCheck(reason is None, reason, metrics)
