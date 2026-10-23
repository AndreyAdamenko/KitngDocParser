"""Provider-wide admission for one asyncio runtime; no credential values are logged."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import random
import time
import weakref

class ProviderCooldown(Exception):
    def __init__(self, seconds):
        self.retry_after = max(0.0, seconds)
        super().__init__("Provider admission deferred")

class ProviderAdmission:
    def __init__(self, max_in_flight=1, *, clock=time.monotonic):
        self._semaphore = asyncio.Semaphore(max(1, max_in_flight))
        self._clock = clock
        self._next_allowed = 0.0
        self._failures = 0
        self._epoch = 0
        self._probe = False

    @property
    def retry_after(self):
        return max(0.0, self._next_allowed - self._clock())

    def failed(self, retry_after=None, base=2.0):
        self._failures += 1
        self._epoch += 1
        delay = retry_after if retry_after is not None else random.uniform(
            max(0.0, base), min(30.0, max(base, base * 2 ** min(self._failures - 1, 10))))
        self._next_allowed = max(self._next_allowed, self._clock() + max(0.0, delay))
        return self.retry_after

    def succeeded(self, started_epoch=None):
        # An older in-flight success cannot erase another request's Retry-After.
        if (started_epoch is not None and started_epoch != self._epoch) or self.retry_after:
            return
        self._failures = 0
        self._next_allowed = 0.0

    @asynccontextmanager
    async def slot(self, budget=None):
        if self.retry_after or self._probe:
            raise ProviderCooldown(self.retry_after or 0.1)
        acquired = False
        probe = False
        try:
            if budget is None:
                await self._semaphore.acquire()
            else:
                budget.check()
                await asyncio.wait_for(self._semaphore.acquire(), budget.remaining())
            acquired = True
            if self.retry_after or self._probe:
                raise ProviderCooldown(self.retry_after or 0.1)
            probe = self._failures > 0
            if probe:
                self._probe = True
            if budget:
                budget.check()
            yield self._epoch
        finally:
            if probe:
                self._probe = False
            if acquired:
                self._semaphore.release()

# Separate loops must not share loop-bound semaphores. This registry coordinates all
# clients with the same account/model in a process/loop, not separate containers.
_loops = weakref.WeakKeyDictionary()
def shared_admission(api_key, model, base_url, max_in_flight=1):
    loop = asyncio.get_running_loop()
    key = hashlib.sha256((api_key + "\0" + model + "\0" + base_url).encode()).digest()
    controllers = _loops.setdefault(loop, {})
    return controllers.setdefault(key, ProviderAdmission(max_in_flight))
