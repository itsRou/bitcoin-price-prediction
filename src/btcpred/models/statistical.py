"""Classical statistical time-series models (Tier 1).

Unlike the row-wise classical/boosting models, these forecast an entire test block at
once from the end of the training series -- the natural fit for ARIMA-family and
exponential-smoothing models. This assumes each walk-forward fold's test set is a single
contiguous block immediately following train (true for `PurgedWalkForwardSplit`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pmdarima as pm
from arch import arch_model
from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR

RANDOM_STATE = 42
VolModel = Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"]


class AutoArimaRegressor(BaseEstimator, RegressorMixin):
    """ARIMA (or ARIMAX, if `exog_columns` is given) via pmdarima's stepwise auto_arima."""

    def __init__(self, exog_columns: tuple[str, ...] | None = None, seasonal: bool = False) -> None:
        self.exog_columns = exog_columns
        self.seasonal = seasonal

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AutoArimaRegressor:
        exog = X[list(self.exog_columns)] if self.exog_columns else None
        self.model_ = pm.auto_arima(
            y, X=exog, seasonal=self.seasonal, suppress_warnings=True, error_action="ignore"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        exog = X[list(self.exog_columns)] if self.exog_columns else None
        return np.asarray(self.model_.predict(n_periods=len(X), X=exog))


class HoltWintersRegressor(BaseEstimator, RegressorMixin):
    """Holt-Winters exponential smoothing, forecasting the test block from train's tail."""

    def __init__(self, trend: str | None = "add", seasonal: str | None = None) -> None:
        self.trend = trend
        self.seasonal = seasonal

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HoltWintersRegressor:
        self.model_ = ExponentialSmoothing(
            y.to_numpy(),
            trend=self.trend,
            seasonal=self.seasonal,
            initialization_method="estimated",
        ).fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.forecast(len(X)))


class ProphetRegressor(BaseEstimator, RegressorMixin):
    """Facebook Prophet, forecasting the test block from the end of the training series."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ProphetRegressor:
        train_df = pd.DataFrame({"ds": y.index.tz_localize(None), "y": y.to_numpy()})
        self.model_ = Prophet(
            daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True
        )
        self.model_.fit(train_df)
        self.last_train_date_ = y.index[-1]
        self.freq_ = pd.infer_freq(y.index) or "D"
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        future_dates = pd.date_range(
            self.last_train_date_.tz_localize(None), periods=len(X) + 1, freq=self.freq_
        )[1:]
        forecast = self.model_.predict(pd.DataFrame({"ds": future_dates}))
        return np.asarray(forecast["yhat"].to_numpy())


class VARRegressor(BaseEstimator, RegressorMixin):
    """Vector autoregression over the target plus a small set of exogenous columns."""

    def __init__(self, exog_columns: tuple[str, ...] = (), max_lags: int = 5) -> None:
        self.exog_columns = exog_columns
        self.max_lags = max_lags

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VARRegressor:
        joint = (
            X[list(self.exog_columns)].copy() if self.exog_columns else pd.DataFrame(index=y.index)
        )
        joint.insert(0, "target", y.to_numpy())
        joint = joint.dropna()
        self.target_col_index_ = 0
        self.model_ = VAR(joint.to_numpy()).fit(maxlags=self.max_lags)
        self.last_values_ = joint.to_numpy()[-self.model_.k_ar :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        forecast = self.model_.forecast(self.last_values_, steps=len(X))
        return np.asarray(forecast[:, self.target_col_index_])


class GARCHVolatilityForecaster(BaseEstimator, RegressorMixin):
    """GARCH(1,1)/EGARCH conditional-volatility forecaster.

    Note: this predicts forecasted *volatility*, not the log-return target itself, so it
    should be scored against a realized-volatility series, not `y_reg`. Included for the
    Tier-1 volatility-modeling sweep, not the return-prediction leaderboard.
    """

    def __init__(
        self, vol: VolModel = "GARCH", p: int = 1, q: int = 1, scale: float = 100.0
    ) -> None:
        self.vol = vol
        self.p = p
        self.q = q
        self.scale = scale

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GARCHVolatilityForecaster:
        self.model_ = arch_model(y.to_numpy() * self.scale, vol=self.vol, p=self.p, q=self.q).fit(
            disp="off"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        forecast = self.model_.forecast(horizon=len(X), reindex=False)
        variance = forecast.variance.to_numpy().flatten()
        return np.sqrt(variance) / self.scale
