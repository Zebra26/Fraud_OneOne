"""Model integrity helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable

from .crypto_utils import compute_file_hash

logger = logging.getLogger(__name__)

_HASH_EXTENSIONS = {".pkl", ".pth", ".joblib", ".json", ".onnx"}
_MANIFEST_DEFAULT = Path("models/model_hashes.json")


def _iter_artifacts(directory: Path) -> Iterable[Path]:
    for item in directory.glob("*"):
        if item.is_file() and item.suffix.lower() in _HASH_EXTENSIONS:
            yield item


def compute_hashes(directory: str | os.PathLike[str]) -> Dict[str, str]:
    """Compute hashes for all artifacts in *directory*.

    Returns a mapping {filename: sha256}.
    """
    dir_path = Path(directory)
    hashes: Dict[str, str] = {}
    for artifact in _iter_artifacts(dir_path):
        try:
            digest = compute_file_hash(artifact)
            hashes[artifact.name] = digest
            logger.info("Computed hash for %s: %s", artifact.name, digest)
        except Exception as exc:  # pragma: no cover - error handling path
            logger.error("Cannot compute hash: %s (%s)", artifact, exc)
    return hashes


def write_hash_manifest(hashes: Dict[str, str], path: str | os.PathLike[str] = _MANIFEST_DEFAULT) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    logger.info("Wrote hash manifest: %s", manifest_path)


def verify_hashes(manifest_path: str | os.PathLike[str] = _MANIFEST_DEFAULT) -> list[str]:
    path = Path(manifest_path)
    if not path.exists():
        logger.warning("Hash manifest missing: %s", path)
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for filename, expected in manifest.items():
        file_path = path.parent / filename
        if not file_path.exists():
            logger.warning("Model integrity mismatch: missing %s", file_path)
            mismatches.append(filename)
            continue
        actual = compute_file_hash(file_path)
        if actual != expected:
            logger.warning("Model integrity mismatch for %s", file_path)
            mismatches.append(filename)
    return mismatches


def verify_on_startup(manifest_path: str | os.PathLike[str] = _MANIFEST_DEFAULT) -> None:
    mismatches = verify_hashes(manifest_path)
    if mismatches:
        logger.warning("Integrity verification failed for: %s", ", ".join(mismatches))
    else:
        logger.info("Model integrity verification passed")



def load_hash_manifest(path: str | os.PathLike[str] = _MANIFEST_DEFAULT) -> Dict[str, str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding='utf-8'))
