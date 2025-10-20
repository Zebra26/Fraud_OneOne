from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json

DEFAULT_WEIGHTS = {"tabular": 0.7, "graph": 0.1, "autoencoder": 0.15, "lstm": 0.05}
DEFAULT_THRESHOLD = 0.7



@dataclass
class ComponentScore:
    name: str
    value: float
    weight: float


class EnsembleDetector:
    """Blend multiple model components into a single fraud probability."""

    def __init__(self, metadata_path: Optional[Path] = None):
        self.components: List[ComponentScore] = []
        self.weights: Dict[str, float] = DEFAULT_WEIGHTS.copy()
        self.threshold: float = DEFAULT_THRESHOLD
        if metadata_path and metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.weights.update(data.get("weights", {}))
            self.threshold = data.get("optimal_threshold", self.threshold)

    def add_component(self, name: str, value: float, weight: Optional[float] = None) -> None:
        weight = weight if weight is not None else self.weights.get(name, DEFAULT_WEIGHTS.get(name, 0.0))
        self.components.append(ComponentScore(name=name, value=value, weight=weight))

    def combined_score(self) -> float:
        if not self.components:
            return 0.0
        weighted = sum(c.value * c.weight for c in self.components)
        total = sum(c.weight for c in self.components)
        if total == 0:
            return 0.0
        return weighted / total

    def as_dict(self) -> Dict[str, float]:
        return {c.name: c.value for c in self.components}

