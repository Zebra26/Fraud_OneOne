from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .model_base import ModelBase

logger = logging.getLogger(__name__)


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderModel(ModelBase):
    """Wrapper for the fraud autoencoder (PyTorch) and its scaler."""

    def __init__(
        self,
        model_path: Path | str = Path("/workspace/models/fraud_autoencoder.pth"),
        scaler_path: Path | str = Path("/workspace/models/autoencoder_scaler.pkl"),
        device: str | torch.device = "cpu",
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.device = torch.device(device)
        self.autoencoder: SimpleAutoencoder | None = None
        self.scaler = None

    def load(self) -> None:
        if self.scaler_path.exists():
            with self.scaler_path.open("rb") as f:
                self.scaler = pickle.load(f)
        else:
            logger.warning("Autoencoder scaler missing at %s", self.scaler_path)

        input_dim = None
        if hasattr(self.scaler, "mean_"):
            input_dim = len(self.scaler.mean_)

        if self.model_path.exists():
            if input_dim is None:
                raise RuntimeError("Scaler required to determine autoencoder input dimensionality")
            self.autoencoder = SimpleAutoencoder(input_dim)
            state = torch.load(self.model_path, map_location=self.device)
            self.autoencoder.load_state_dict(state)
            self.autoencoder.to(self.device)
            self.autoencoder.eval()
        else:
            logger.warning("Autoencoder weights missing at %s", self.model_path)

    def score(self, features: Iterable[float]) -> float:
        if self.autoencoder is None or self.scaler is None:
            return 0.0

        array = np.asarray(list(features), dtype=np.float32)
        scaled = self.scaler.transform([array])[0]
        tensor = torch.from_numpy(scaled).to(self.device)
        with torch.no_grad():
            reconstruction = self.autoencoder(tensor)
            loss = torch.mean((tensor - reconstruction) ** 2).item()
        return float(loss)

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if self.scaler is not None:
            with (target / "autoencoder_scaler.pkl").open("wb") as f:
                pickle.dump(self.scaler, f)
        if self.autoencoder is not None:
            torch.save(self.autoencoder.state_dict(), target / "fraud_autoencoder.pth")
