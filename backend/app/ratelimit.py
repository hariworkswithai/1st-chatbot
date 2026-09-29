"""Minimal app-level (per-IP) rate limiter: sliding window in memory.

This is *separate* from the provider's 429 handling in the LLM layer.
Purpose: stop one client from hammering our backend / burning API budget.
No Redis — overkill for V1.
"""

from __future__ import annotations

import time
from collections import deque


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        bucket = self._hits.setdefault(key, deque())
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) < self.max_requests:
            bucket.append(now)
            return True, 0.0
        retry_after = self.window - (now - bucket[0])
        return False, max(0.0, retry_after)

    def reset(self) -> None:
        self._hits.clear()
