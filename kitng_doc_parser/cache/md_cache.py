"""Долговременный кэш Markdown с авто-инвалидацией по mtime и size."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..models import CacheEntry


class MdCache:
    """Файловый кэш Markdown-документов с метаданными и проверкой свежести."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, item_id: str) -> str:
        # Для безопасности путей хешируем относительный путь, сохраняя исходное имя в конце
        safe_name = Path(item_id).name
        stem = safe_name[:40] if safe_name else "doc"
        h = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]
        return f"{h}_{stem}"

    def _dir(self, scope: str) -> Path:
        # Очищаем scope от спецсимволов
        safe_scope = "".join(c for c in scope if c.isalnum() or c in ("-", "_", "."))
        return self._root / safe_scope

    def _md_path(self, scope: str, item_id: str) -> Path:
        return self._dir(scope) / f"{self._hash_key(item_id)}.md"

    def _meta_path(self, scope: str, item_id: str) -> Path:
        return self._dir(scope) / f"{self._hash_key(item_id)}.meta.json"

    def get_meta(self, scope: str, item_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self._meta_path(scope, item_id)
        if not meta_path.is_file():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def get(
        self,
        scope: str,
        item_id: str,
        mtime: Optional[float] = None,
        size_bytes: Optional[int] = None,
    ) -> Optional[CacheEntry]:
        """Возвращает запись из кэша. Если mtime или size_bytes изменились, возвращает None."""
        meta = self.get_meta(scope, item_id)
        if meta is None:
            return None

        # Проверка актуальности по mtime
        if mtime is not None:
            cached_mtime = meta.get("mtime")
            if cached_mtime is not None and mtime > (cached_mtime + 1.0):
                # Файл на диске новее кэша более чем на 1 секунду — кэш устарел!
                return None

        # Проверка актуальности по размеру
        if size_bytes is not None:
            cached_size = meta.get("size_bytes")
            if cached_size is not None and cached_size != size_bytes:
                return None

        md_path = self._md_path(scope, item_id)
        if not md_path.is_file():
            return None

        try:
            markdown = md_path.read_text(encoding="utf-8")
        except OSError:
            return None

        return CacheEntry(markdown=markdown, meta=meta)

    def save(
        self,
        scope: str,
        item_id: str,
        markdown: str,
        meta: Dict[str, Any],
    ) -> None:
        """Атомарно сохраняет Markdown и метаданные."""
        target_dir = self._dir(scope)
        target_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self._md_path(scope, item_id), markdown)
        self._atomic_write(
            self._meta_path(scope, item_id),
            json.dumps(meta, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
