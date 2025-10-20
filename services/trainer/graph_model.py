from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from .model_base import ModelBase

logger = logging.getLogger(__name__)


class GraphModel(ModelBase):
    """Wrapper around pre-trained Node2Vec embeddings and downstream classifier.

    Expected artifacts:
      - node2vec_model.pkl : Dict[str, np.ndarray] mapping node ids to embeddings
      - optimized_fraud_detection_system.pkl : downstream classifier (joblib/pickle)
    """

    def __init__(
        self,
        embedding_path: Path | str = Path("/workspace/models/node2vec_model.pkl"),
        classifier_path: Path | str = Path("/workspace/models/optimized_fraud_detection_system.pkl"),
    ):
        self.embedding_path = Path(embedding_path)
        self.classifier_path = Path(classifier_path)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.classifier = None

    def load(self) -> None:
        if self.embedding_path.exists():
            with self.embedding_path.open("rb") as f:
                self.embeddings = pickle.load(f)
        else:  # graceful degradation
            self.embeddings = {}
            logger.warning("Node2Vec embeddings missing at %s", self.embedding_path)

        if self.classifier_path.exists():
            with self.classifier_path.open("rb") as f:
                self.classifier = pickle.load(f)
        else:
            logger.warning("Graph classifier missing at %s", self.classifier_path)

    def score(self, graph_context: Mapping[str, str] | Tuple[str, str]) -> float:
        """Return a fraud score based on graph proximity.

        If classifier is provided, feed concatenated embeddings.
        Otherwise return cosine similarity heuristic (0..1).
        """

        if not self.embeddings:
            return 0.0

        if isinstance(graph_context, tuple):
            source_id, target_id = graph_context
        else:
            source_id = graph_context.get("source_id")
            target_id = graph_context.get("target_id")

        if source_id is None or target_id is None:
            return 0.0

        emb_src = self.embeddings.get(str(source_id))
        emb_dst = self.embeddings.get(str(target_id))
        if emb_src is None or emb_dst is None:
            return 0.0

        if self.classifier is not None:
            features = np.concatenate([emb_src, emb_dst, np.abs(emb_src - emb_dst)])
            prob = float(self.classifier.predict_proba([features])[0][1])
            return prob

        denom = np.linalg.norm(emb_src) * np.linalg.norm(emb_dst)
        if denom == 0:
            return 0.0
        similarity = float(np.dot(emb_src, emb_dst) / denom)
        # Convert similarity to suspiciousness score (higher -> more suspicious)
        return (1 - similarity) * 0.5 + 0.5

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "node2vec_model.pkl").open("wb") as f:
            pickle.dump(self.embeddings, f)
        if self.classifier is not None:
            with (target / "optimized_fraud_detection_system.pkl").open("wb") as f:
                pickle.dump(self.classifier, f)

