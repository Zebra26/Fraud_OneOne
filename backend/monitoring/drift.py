from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prometheus_client import Counter, Gauge


DRIFT_ALERTS = Counter("drift_alerts_total", "Drift alerts total")
_GAUGES: Dict[str, Gauge] = {}


def _get_gauge(feature: str) -> Gauge:
    name = f"psi_feature_{_sanitize(feature)}"
    if name not in _GAUGES:
        _GAUGES[name] = Gauge(name, f"PSI for feature {feature}")
    return _GAUGES[name]


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def compute_psi(observed: List[float], expected: List[float]) -> float:
    """Compute Population Stability Index given observed and expected proportions.

    Both lists must have the same length and represent proportions per bin.
    Adds a small epsilon to avoid division by zero and log issues.
    """
    if len(observed) != len(expected) or not observed:
        return 0.0
    eps = 1e-6
    psi = 0.0
    for o, e in zip(observed, expected):
        o1 = max(o, eps)
        e1 = max(e, eps)
        psi += (o1 - e1) * (math_log(o1 / e1))
    return float(psi)


def math_log(x: float) -> float:
    # Local tiny log to avoid importing math globally in callers
    import math

    return math.log(x)


def load_baseline_from_model_meta(model_dir: Path | str) -> Dict[str, Dict[str, List[float]]]:
    """Load baseline histograms from model_performance.json if present.

    Expected structure inside file (optional):
    {
      "drift_baseline": {
        "feature_name": {"edges": [..bin edges..], "expected": [..proportions..]},
        ...
      }
    }
    Returns an empty dict if not found.
    """
    model_dir = Path(model_dir)
    perf = model_dir / "model_performance.json"
    if not perf.exists():
        return {}
    try:
        data = json.loads(perf.read_text(encoding="utf-8"))
        baseline = data.get("drift_baseline")
        if isinstance(baseline, dict):
            # Basic validation
            cleaned: Dict[str, Dict[str, List[float]]] = {}
            for feat, meta in baseline.items():
                edges = meta.get("edges") if isinstance(meta, dict) else None
                expected = meta.get("expected") if isinstance(meta, dict) else None
                if isinstance(edges, list) and isinstance(expected, list) and len(expected) == max(len(edges) - 1, 0):
                    cleaned[feat] = {"edges": edges, "expected": expected}
            return cleaned
    except Exception:
        return {}
    return {}


class DriftMonitor:
    def __init__(self, baseline: Dict[str, Dict[str, List[float]]], window: int = 5000, warn: float = 0.2, crit: float = 0.3):
        self.baseline = baseline
        self.window = max(window, 1)
        self.warn = warn
        self.crit = crit
        # Observed counts per feature/bin
        self._counts: Dict[str, List[int]] = {feat: [0] * len(meta.get("expected", [])) for feat, meta in baseline.items()}
        self._n: int = 0

    def observe(self, features: Dict[str, float]) -> Dict[str, float]:
        """Update histograms with a single feature snapshot and compute PSI values.

        Returns a mapping feature -> psi (only for features with baseline).
        """
        if not self.baseline:
            return {}
        self._n = min(self._n + 1, self.window)
        # Update counts
        for feat, meta in self.baseline.items():
            val = features.get(feat)
            if val is None:
                continue
            edges = meta["edges"]
            # find bin
            idx = _bin_index(val, edges)
            if 0 <= idx < len(self._counts[feat]):
                self._counts[feat][idx] += 1

        # Compute observed proportions
        psi_map: Dict[str, float] = {}
        for feat, meta in self.baseline.items():
            expected = meta["expected"]
            counts = self._counts[feat]
            total = sum(counts) or 1
            observed = [c / total for c in counts]
            psi = compute_psi(observed, expected)
            psi_map[feat] = psi
            _get_gauge(feat).set(psi)
            if psi >= self.crit:
                DRIFT_ALERTS.inc()
        # Decay window to keep bounded memory: when window reached, scale down counts by 0.9
        if self._n >= self.window:
            for feat in self._counts:
                self._counts[feat] = [max(int(c * 0.9), 0) for c in self._counts[feat]]
            self._n = int(self._n * 0.9)
        return psi_map


def _bin_index(value: float, edges: List[float]) -> int:
    # edges are inclusive on the right for the last bin
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    # last bin
    return len(edges) - 2 if value >= edges[-2] else -1

