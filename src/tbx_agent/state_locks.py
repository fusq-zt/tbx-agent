"""Shared single-process locking for read/modify/write business operations."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock


def state_lock_key(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _LockEntry:
    lock: RLock
    users: int = 0


class ScopedLockPool:
    """Release unused context locks after both holders and waiters leave."""

    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[str, _LockEntry] = {}

    @contextmanager
    def hold(self, key: str):
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _LockEntry(lock=RLock())
                self._entries[key] = entry
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(key, None)
