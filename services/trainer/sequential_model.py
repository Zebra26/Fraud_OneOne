from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .model_base import ModelBase

logger = logging.getLogger(__name__)


class LSTMDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        prob = self.head(last_hidden)
        return prob


class SequentialModel(ModelBase):
    """Wrapper around the LSTM-based sequential fraud detector."""

    def __init__(
        self,
        model_path: Path | str = Path("/workspace/models/lstm_fraud_detector.pth"),
        scaler_path: Path | str = Path("/workspace/models/sequential_scaler.pkl"),
        device: str | torch.device = "cpu",
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.device = torch.device(device)
        self.detector: LSTMDetector | None = None
        self.scaler = None

    def load(self) -> None:
        if self.scaler_path.exists():
            with self.scaler_path.open("rb") as f:
                self.scaler = pickle.load(f)
        else:
            logger.warning("Sequential scaler missing at %s", self.scaler_path)

        input_dim = None
        if hasattr(self.scaler, "mean_"):
            input_dim = len(self.scaler.mean_)

        if self.model_path.exists():
            if input_dim is None:
                raise RuntimeError("Sequential scaler required to determine LSTM input dimensionality")
            self.detector = LSTMDetector(input_dim)
            state = torch.load(self.model_path, map_location=self.device)
            self.detector.load_state_dict(state)
            self.detector.to(self.device)
            self.detector.eval()
        else:
            logger.warning("LSTM weights missing at %s", self.model_path)

    def score(self, sequence: Sequence[Iterable[float]]) -> float:
        if self.detector is None or self.scaler is None:
            return 0.0

        sequence_np = np.asarray([list(vec) for vec in sequence], dtype=np.float32)
        scaled = self.scaler.transform(sequence_np)
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.detector(tensor).item()
        return float(prob)

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if self.scaler is not None:
            with (target / "sequential_scaler.pkl").open("wb") as f:
                pickle.dump(self.scaler, f)
        if self.detector is not None:
            torch.save(self.detector.state_dict(), target / "lstm_fraud_detector.pth")

