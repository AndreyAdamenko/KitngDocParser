"""Управление конкурентным доступом к документам."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict


class DocumentLockRegistry:
    """Реестр asyncio-блокировок для предотвращения одновременного парсинга одного файла."""

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:
        async with self._guard:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            item_lock = self._locks[key]

        async with item_lock:
            yield
