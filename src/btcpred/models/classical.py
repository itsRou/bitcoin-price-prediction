"""Classical ML models (Tier 2), each wrapped in a Pipeline so scaling is fold-local.

`get_regression_models()` / `get_classification_models()` return name -> zero-arg factory
dicts, so a fresh, unfitted pipeline is built per fold rather than one shared, stateful
instance leaking scaler statistics across folds.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42


def _pipeline(model: object) -> Pipeline:
    """Wrap a model in a scaler Pipeline so `.fit` never leaks cross-fold statistics."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def get_regression_models() -> dict[str, Callable[[], Pipeline]]:
    """Factories for every Tier-2 regression model, scaler-wrapped."""
    return {
        "linear": lambda: _pipeline(LinearRegression()),
        "ridge": lambda: _pipeline(Ridge(random_state=RANDOM_STATE)),
        "lasso": lambda: _pipeline(Lasso(random_state=RANDOM_STATE)),
        "elastic_net": lambda: _pipeline(ElasticNet(random_state=RANDOM_STATE)),
        "bayesian_ridge": lambda: _pipeline(BayesianRidge()),
        "huber": lambda: _pipeline(HuberRegressor()),
        "knn": lambda: _pipeline(KNeighborsRegressor()),
        "svr": lambda: _pipeline(SVR()),
        "decision_tree": lambda: _pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
        "random_forest": lambda: _pipeline(RandomForestRegressor(random_state=RANDOM_STATE)),
        "extra_trees": lambda: _pipeline(ExtraTreesRegressor(random_state=RANDOM_STATE)),
        "gradient_boosting": lambda: _pipeline(
            GradientBoostingRegressor(random_state=RANDOM_STATE)
        ),
        "adaboost": lambda: _pipeline(AdaBoostRegressor(random_state=RANDOM_STATE)),
    }


def get_classification_models() -> dict[str, Callable[[], Pipeline]]:
    """Factories for every Tier-2 classification model, scaler-wrapped."""
    return {
        "logistic_regression": lambda: _pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "knn": lambda: _pipeline(KNeighborsClassifier()),
        "svc": lambda: _pipeline(SVC(random_state=RANDOM_STATE)),
        "decision_tree": lambda: _pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "random_forest": lambda: _pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
        "extra_trees": lambda: _pipeline(ExtraTreesClassifier(random_state=RANDOM_STATE)),
        "gradient_boosting": lambda: _pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
        "adaboost": lambda: _pipeline(AdaBoostClassifier(random_state=RANDOM_STATE)),
        "gaussian_nb": lambda: _pipeline(GaussianNB()),
    }
