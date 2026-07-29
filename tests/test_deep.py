"""Tests for the shared windowed dataset and all Tier-3 deep-learning architectures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from btcpred.models.deep import (
    Architecture,
    TorchSequenceRegressor,
    TrainingConfig,
    WindowedSequenceDataset,
)

ALL_ARCHITECTURES: tuple[Architecture, ...] = (
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
)

_FAST_CONFIG = TrainingConfig(max_epochs=3, patience=2, batch_size=8, val_fraction=0.2)


def _make_xy(n: int = 150) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(0.3 * X["f1"] - 0.1 * X["f2"] + rng.normal(scale=0.05, size=n))
    return X, y


def test_windowed_dataset_produces_expected_length_and_shapes() -> None:
    X = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    y = np.arange(20, dtype=np.float32)
    window = 5

    dataset = WindowedSequenceDataset(X, y, window)
    seq, target = dataset[0]

    assert len(dataset) == 20 - window + 1
    assert seq.shape == (window, 3)
    assert target.item() == y[window - 1]


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_each_architecture_fits_and_predicts(architecture: Architecture) -> None:
    X, y = _make_xy()
    train_X, train_y = X.iloc[:120], y.iloc[:120]
    test_X = X.iloc[120:]

    model = TorchSequenceRegressor(
        architecture=architecture, window=10, hidden_size=8, config=_FAST_CONFIG
    ).fit(train_X, train_y)
    preds = model.predict(test_X)

    assert len(preds) == len(test_X)
    assert np.all(np.isfinite(preds))


def test_predict_pads_when_test_block_is_shorter_than_window() -> None:
    X, y = _make_xy()
    train_X, train_y = X.iloc[:120], y.iloc[:120]
    test_X = X.iloc[120:125]  # shorter than window=10

    model = TorchSequenceRegressor(
        architecture="lstm", window=10, hidden_size=8, config=_FAST_CONFIG
    ).fit(train_X, train_y)
    preds = model.predict(test_X)

    assert len(preds) == len(test_X)


def test_seed_gives_reproducible_predictions() -> None:
    X, y = _make_xy()
    train_X, train_y = X.iloc[:120], y.iloc[:120]
    test_X = X.iloc[120:]

    torch.use_deterministic_algorithms(False)
    model_a = TorchSequenceRegressor(
        architecture="mlp", window=10, hidden_size=8, config=_FAST_CONFIG, seed=7
    ).fit(train_X, train_y)
    model_b = TorchSequenceRegressor(
        architecture="mlp", window=10, hidden_size=8, config=_FAST_CONFIG, seed=7
    ).fit(train_X, train_y)

    np.testing.assert_allclose(model_a.predict(test_X), model_b.predict(test_X), atol=1e-5)
