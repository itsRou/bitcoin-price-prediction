"""Purged, embargoed, expanding-window walk-forward cross-validation splitter."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class PurgedWalkForwardSplit:
    """Expanding-window walk-forward CV with purge and embargo around each test fold.

    The data is divided into `n_splits + 1` contiguous, chronologically ordered blocks.
    Fold i's test set is block i; its train set is every row before the test set, with
    two safeguards against leakage from overlapping forward-looking targets:

    - purge: the `purge` rows immediately before each test set are dropped from training,
      since their targets (which look `purge` bars ahead) may reach into the test period.
    - embargo: the `embargo` rows immediately after every *prior* test set are permanently
      excluded from training in all subsequent folds, since those rows' features may still
      reflect information from around that test period.

    Both should typically be set to the largest forecast horizon `h` in use.
    """

    def __init__(self, n_splits: int = 5, purge: int = 1, embargo: int = 1) -> None:
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def split(
        self, X: pd.DataFrame, y: pd.Series | None = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) positional-index pairs, fold by fold."""
        n_samples = len(X)
        fold_bounds = np.array_split(np.arange(n_samples), self.n_splits + 1)

        for fold in range(1, self.n_splits + 1):
            test_idx = fold_bounds[fold]
            if len(test_idx) == 0:
                continue

            train_end = max(test_idx[0] - self.purge, 0)
            train_idx = np.arange(0, train_end)

            for prior_fold in range(1, fold):
                prior_test_end = fold_bounds[prior_fold][-1]
                embargo_start = prior_test_end + 1
                embargo_end = prior_test_end + self.embargo
                train_idx = train_idx[(train_idx < embargo_start) | (train_idx > embargo_end)]

            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(
        self, X: pd.DataFrame | None = None, y: pd.Series | None = None, groups: object = None
    ) -> int:
        """Return the number of splits, matching the sklearn splitter interface."""
        return self.n_splits
