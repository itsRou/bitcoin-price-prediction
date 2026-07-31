"""Tests for the purged, embargoed, expanding-window walk-forward splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcpred.validation.splitters import PurgedWalkForwardSplit


def _make_frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(n)})


def test_get_n_splits_matches_constructor_arg() -> None:
    splitter = PurgedWalkForwardSplit(n_splits=4, purge=1, embargo=1)
    assert splitter.get_n_splits() == 4


def test_train_always_precedes_test_chronologically() -> None:
    splitter = PurgedWalkForwardSplit(n_splits=4, purge=2, embargo=2)
    X = _make_frame(100)

    for train_idx, test_idx in splitter.split(X):
        assert train_idx.max() < test_idx.min()


def test_purge_removes_rows_immediately_before_test() -> None:
    purge = 3
    splitter = PurgedWalkForwardSplit(n_splits=4, purge=purge, embargo=0)
    X = _make_frame(100)

    for train_idx, test_idx in splitter.split(X):
        assert test_idx[0] - train_idx.max() - 1 == purge


def test_embargo_excludes_rows_after_prior_test_folds() -> None:
    embargo = 3
    splitter = PurgedWalkForwardSplit(n_splits=4, purge=0, embargo=embargo)
    X = _make_frame(100)

    folds = list(splitter.split(X))
    first_test_end = folds[0][1][-1]
    embargoed_rows = set(range(first_test_end + 1, first_test_end + 1 + embargo))

    _, second_train_idx = folds[1][0], folds[1][0]
    assert embargoed_rows.isdisjoint(set(second_train_idx))


def test_yields_expected_number_of_folds() -> None:
    splitter = PurgedWalkForwardSplit(n_splits=5, purge=1, embargo=1)
    X = _make_frame(200)

    folds = list(splitter.split(X))
    assert len(folds) == 5
