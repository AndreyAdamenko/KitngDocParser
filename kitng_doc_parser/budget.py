"""A cumulative budget shared by admission, transport and PDF work."""
from dataclasses import dataclass
from typing import Callable, Optional
import asyncio
import time

@dataclass
class ParseBudget:
    deadline: float
    max_new_chunks: int = 1
    max_attempts: int = 1
    cancelled: Optional[Callable[[], bool]] = None

    @classmethod
    def seconds(cls, seconds: float, **kwargs):
        return cls(time.monotonic() + max(0.0, seconds), **kwargs)

    def remaining(self) -> float:
        self.check_cancelled()
        return max(0.0, self.deadline - time.monotonic())

    def check_cancelled(self):
        if self.cancelled is not None and self.cancelled():
            raise asyncio.CancelledError()

    def check(self):
        self.check_cancelled()
        if self.remaining() <= 0:
            raise TimeoutError("Parse budget exhausted")
