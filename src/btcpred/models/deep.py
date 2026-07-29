"""PyTorch deep-learning models (Tier 3), sharing one windowed dataset and training loop.

Every architecture below is deliberately small: a `forward()` method operating on a
(batch, window, n_features) tensor. `TorchSequenceRegressor` handles everything an
architecture shouldn't have to repeat -- scaling, windowing, the chronological
early-stopping split, gradient clipping, LR scheduling, seeding, and CPU/GPU placement --
so adding a new architecture only ever means adding a new `nn.Module` and one branch in
`_build_architecture`.

Scope note: `neuralforecast`-based N-HiTS/TFT are explicitly optional in the project plan
and are skipped here to avoid a heavy extra dependency for marginal benchmark coverage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

Architecture = Literal[
    "mlp",
    "rnn",
    "lstm",
    "gru",
    "bilstm",
    "cnn1d",
    "cnn_lstm",
    "lstm_attention",
    "transformer",
    "tcn",
]
RANDOM_STATE = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowedSequenceDataset(Dataset):
    """Sliding windows of features ending at row t, paired with the target at row t."""

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int) -> None:
        self.X = X
        self.y = y
        self.window = window

    def __len__(self) -> int:
        return max(len(self.X) - self.window + 1, 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.X[idx : idx + self.window]
        target = self.y[idx + self.window - 1]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class _MLP(nn.Module):
    def __init__(self, input_size: int, window: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _RecurrentNet(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, cell: str = "lstm", bidirectional: bool = False
    ) -> None:
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class _CNN1D(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.head(features).squeeze(-1)


class _CNNLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        output, _ = self.lstm(conv_out)
        return self.head(output[:, -1, :]).squeeze(-1)


class _LSTMAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        scores = self.attn(output).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (output * weights).sum(dim=1)
        return self.head(context).squeeze(-1)


class _TransformerEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, window: int, n_heads: int = 4, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, window, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_proj(x) + self.pos_embedding)
        return self.head(encoded.mean(dim=1)).squeeze(-1)


class _TemporalBlock(nn.Module):
    """One causal, dilated 1D convolution block (trims the right-side padding)."""

    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, : -self.padding] if self.padding else out
        return self.relu(out)


class _TCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_levels: int = 3) -> None:
        super().__init__()
        layers = []
        in_channels = input_size
        for level in range(n_levels):
            layers.append(_TemporalBlock(in_channels, hidden_size, dilation=2**level))
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x.transpose(1, 2))
        return self.head(features[:, :, -1]).squeeze(-1)


def _build_architecture(
    name: Architecture, input_size: int, window: int, hidden_size: int
) -> nn.Module:
    if name == "mlp":
        return _MLP(input_size, window, hidden_size)
    if name in ("rnn", "lstm", "gru"):
        return _RecurrentNet(input_size, hidden_size, cell=name)
    if name == "bilstm":
        return _RecurrentNet(input_size, hidden_size, cell="lstm", bidirectional=True)
    if name == "cnn1d":
        return _CNN1D(input_size, hidden_size)
    if name == "cnn_lstm":
        return _CNNLSTM(input_size, hidden_size)
    if name == "lstm_attention":
        return _LSTMAttention(input_size, hidden_size)
    if name == "transformer":
        return _TransformerEncoder(input_size, hidden_size, window)
    if name == "tcn":
        return _TCN(input_size, hidden_size)
    raise ValueError(f"Unknown architecture: {name}")


@dataclass
class TrainingConfig:
    """Shared training hyperparameters for every Tier-3 architecture."""

    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 32
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.15


class TorchSequenceRegressor(BaseEstimator, RegressorMixin):
    """Uniform sklearn-style wrapper around every Tier-3 architecture.

    Note on `.predict(X)`: sklearn's per-row-CV interface only ever hands this method the
    rows to predict, with no preceding context. Since every architecture here needs
    `window` rows of history to predict even the first row, the first `window - 1` rows
    of any `predict(X)` call are padded by repeating the first available row -- a
    documented approximation, not a claim that those particular predictions are as
    reliable as the rest.
    """

    def __init__(
        self,
        architecture: Architecture = "lstm",
        window: int = 30,
        hidden_size: int = 64,
        config: TrainingConfig | None = None,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.architecture = architecture
        self.window = window
        self.hidden_size = hidden_size
        self.config = config
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TorchSequenceRegressor:
        cfg = self.config or TrainingConfig()
        _set_seed(self.seed)
        device = _get_device()

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X.to_numpy()).astype(np.float32)
        y_values = y.to_numpy(dtype=np.float32)

        split_at = max(int(len(X_scaled) * (1 - cfg.val_fraction)), self.window + 1)
        train_ds = WindowedSequenceDataset(X_scaled[:split_at], y_values[:split_at], self.window)
        val_start = max(split_at - self.window, 0)
        val_ds = WindowedSequenceDataset(X_scaled[val_start:], y_values[val_start:], self.window)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        self.model_ = _build_architecture(
            self.architecture, X.shape[1], self.window, self.hidden_size
        ).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for _epoch in range(cfg.max_epochs):
            self.model_.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(batch_X), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            val_loss = self._evaluate(val_loader, loss_fn, device)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.device_ = device
        self.config_ = cfg
        return self

    def _evaluate(self, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
        self.model_.eval()
        losses = []
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                losses.append(loss_fn(self.model_(batch_X), batch_y).item())
        return float(np.mean(losses)) if losses else float("inf")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler_.transform(X.to_numpy()).astype(np.float32)
        pad = np.repeat(X_scaled[:1], self.window - 1, axis=0)
        X_padded = np.vstack([pad, X_scaled])

        dataset = WindowedSequenceDataset(
            X_padded, np.zeros(len(X_padded), dtype=np.float32), self.window
        )
        loader = DataLoader(dataset, batch_size=self.config_.batch_size, shuffle=False)

        self.model_.eval()
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                predictions.append(self.model_(batch_X.to(self.device_)).cpu().numpy())
        return np.concatenate(predictions) if predictions else np.array([])
