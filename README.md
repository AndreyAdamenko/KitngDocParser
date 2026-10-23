# kitng-doc-parser

Общая библиотека конвейера парсинга документов для MCP-серверов института:
- `KitngProjectsFolderMCP` (файлы сетевого диска `\\kitng-fs\F\04_projects`)
- `VitrocadMCP` (документы и договоры VitroCAD)
- `PostKitngMcp` (входящая/исходящая корреспонденция)

## Возможности
- **DOCX / DOCM**: Локальное извлечение заголовков, текста и таблиц через `python-docx`.
- **XLSX / XLS**: Локальное извлечение листов и таблиц в Markdown через `openpyxl`.
- **TXT / CSV / LOG**: Прямое чтение с авто-определением кодировок (UTF-8, CP1251).
- **PDF**:
  - Быстрый текстовый слой через `pypdf`.
  - Чанковый OCR через OpenRouter / Gemini API для сканов.
- **Интеллектуальный кэш**:
  - Сохранение в Markdown с метаданными.
  - Автоматическая инвалидация по дате изменения файла (`mtime`) и размеру (`size_bytes`).
  - Чанковый кэш OCR для возобновляемого постраничного парсинга.

## Установка в режиме разработки
```bash
pip install -e C:\Users\a.adamenko\source\repos\KitngDocParser
```

## PDF policy 1.2

PDF processing uses one native-text inspection pass in a disposable subprocess
when a `ParseBudget` is supplied. All usable native pages are checkpointed in
that slice. The budget's `max_new_chunks` limits **paid OCR pages**, not native
pages. PDF OCR always sends one page per request; legacy `chunk_size` and
`chunk_threshold` constructor arguments remain accepted for caller compatibility.
Without an explicit budget the caller retains control of overall parse duration.

A native page needs at least 40 visible / 20 alphanumeric characters, no excessive
replacement/private-use characters, valid bounded content, and no significant
raster content (20% displayed page area) or uncertain image analysis. These are
conservative extraction checks, not verification of reading order, table layout,
signatures or completeness. In particular native text is not presented as OCR.

`ParseResult` preserves its existing fields and adds `reason_code`, `native_pages`,
`ocr_pages`, `page_origins`, and `quality_policy`. `pages`, `total_chunks`, and
`completed_chunks` refer to source pages / per-page checkpoints. Complete origins
are `native_pdf`, `mixed_pdf`, or `ocr`; a complete native or mixed document must
not be reported as entirely OCR. Unavailable pages remain explicit comments in
a partial document; pypdf text is never silently substituted for an OCR failure.

OCR responses are validated **before** return and checkpoint `done`: only a
normal `stop` finish reason is accepted; truncated/filtered/missing finish reasons,
empty output, pathological repeated characters/leaders and empty table loops
are rejected. Bounds are 16,384 output tokens, 65,536 characters per page and
1 MiB of HTTP response bytes. The whole-document quality API caps content at
16 Mi characters. These are operational safety ceilings, not expected output
lengths or semantic quality scores; dense legitimate output beyond them needs an
explicitly reviewed policy. No validator rewrites numbers or table cells, and
no rejected result is silently truncated and accepted. The prompt compacts visual
dot leaders while preserving section labels and page references.

`validate_markdown_quality(text, *, finish_reason=None, page_count=None,
max_chars=None, require_finish_reason=False)` returns `QualityCheck(valid,
reason_code, metrics)` for cached whole-document checks **before pagination**.
Passing the checks does not certify content accuracy. Validate cached Markdown
and snapshots in the caller too; an already assembled old MdCache is separate
from this library's per-page checkpoint cache.

The processing revision includes exact source bytes and the quality/native-first
policy. Previous revisions are not reused. Checkpoint reads and writes also run
the deterministic validator, so invalid old checkpoint files return no usable
content without deleting or overwriting the original files on a read.

HTTP 429 without a valid `Retry-After` defers this credential/model/provider
controller for 60 seconds. An explicit header is respected. Transport failures
keep their separate backoff; waiting does not extend a supplied cumulative
budget. Shared admission is process/event-loop scoped, so callers using several
containers need their own shared coordination. A nonretryable `invalid_content`
needs investigation or an explicitly selected alternative, not blind resubmission.
