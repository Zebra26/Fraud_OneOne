from __future__ import annotations

from pathlib import Path
from typing import Iterable


class ModelBase:
    """Abstract helper for trainer model wrappers."""

    def load(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, data: Iterable[float]) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def save(self, path: Path | str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

