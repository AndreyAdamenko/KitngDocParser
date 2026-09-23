"""Чанковый кэш OCR для возобновляемого постраничного парсинга PDF."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


class ChunkCache:
    """Хранит промежуточные результаты OCR по чанкам страниц."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root) / "chunks"
        self._root.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, item_id: str) -> str:
        return hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]

    def _doc_dir(self, scope: str, item_id: str) -> Path:
        safe_scope = "".join(c for c in scope if c.isalnum() or c in ("-", "_", "."))
        return self._root / safe_scope / self._hash_key(item_id)

    def get_manifest(self, scope: str, item_id: str) -> Optional[Dict[str, Any]]:
        path = self._doc_dir(scope, item_id) / "manifest.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_manifest(self, scope: str, item_id: str, manifest: Dict[str, Any]) -> None:
        doc_dir = self._doc_dir(scope, item_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(
            doc_dir / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    def get_chunk(self, scope: str, item_id: str, chunk_index: int) -> Optional[str]:
        path = self._doc_dir(scope, item_id) / f"chunk_{chunk_index}.md"
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def save_chunk(self, scope: str, item_id: str, chunk_index: int, markdown: str) -> None:
        doc_dir = self._doc_dir(scope, item_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(doc_dir / f"chunk_{chunk_index}.md", markdown)

    def clear(self, scope: str, item_id: str) -> None:
        doc_dir = self._doc_dir(scope, item_id)
        if not doc_dir.is_dir():
            return
        for child in doc_dir.glob("*"):
            try:
                child.unlink()
            except OSError:
                pass
        try:
            doc_dir.rmdir()
        except OSError:
            pass

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
