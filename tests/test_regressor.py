"""Tests for regressor cross-validated estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, train_test_split

from tree_machine import (
    QuantileCV,
    RegressionCV,
    RegressionCVConfig,
    default_regression,
)


@pytest.fixture(scope="session")
def regression_data():
    """Return a regression train/test split as pandas DataFrames."""
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=20)
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(20)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def trained_model(regression_data):
    """Fit a default RegressionCV model."""
    X_train, _, y_train, _ = regression_data

    model = RegressionCV(
        metric="mse",
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=default_regression,
    )
    model.fit(X_train, y_train)

    return model


@pytest.fixture(scope="session")
def trained_quantile(regression_data):
    """Fit a default QuantileCV model."""
    X_train, _, y_train, _ = regression_data

    model = QuantileCV(
        alpha=0.45,
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=default_regression,
    )
    model.fit(X_train, y_train)

    return model


def test_model_predict(regression_data, trained_model):
    """predict should return finite numeric predictions."""
    _, X_test, _, _ = regression_data
    assert all(np.isreal(trained_model.predict(X_test)))


def test_model_predict_quantile(regression_data, trained_quantile):
    """Quantile predictor should return finite numeric predictions."""
    _, X_test, _, _ = regression_data
    assert all(np.isreal(trained_quantile.predict(X_test)))


def test_model_score(regression_data, trained_model):
    """score should return a truthy float for a fitted model."""
    _, X_test, _, y_test = regression_data
    assert trained_model.score(X_test, y_test)


def test_model_explain(regression_data, trained_model):
    """explain should return SHAP values with expected shape."""
    _, X_test, _, _ = regression_data

    explain = trained_model.explain(X_test)
    assert explain["shap_values"].shape == (250, 20)


def test_model_performance(regression_data, trained_model):
    """Trained model should outperform a dummy baseline."""
    X_train, X_test, y_train, y_test = regression_data

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)

    baseline_score = dummy.score(X_test, y_test)
    model_score = trained_model.score(X_test, y_test)

    assert baseline_score < model_score


@pytest.fixture(scope="session")
def trained_model_catboost(regression_data):
    """Fit a RegressionCV model using the CatBoost backend."""
    X_train, _, y_train, _ = regression_data

    config = RegressionCVConfig(
        monotone_constraints={},
        interactions=[],
        n_jobs=1,
        parameters=default_regression.parameters,
        return_train_score=True,
    )

    model = RegressionCV(
        metric="mse",
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=config,
        backend="catboost",
    )
    model.fit(X_train, y_train)

    return model


@pytest.fixture(scope="session")
def trained_quantile_catboost(regression_data):
    """Fit a QuantileCV model using the CatBoost backend."""
    X_train, _, y_train, _ = regression_data

    config = RegressionCVConfig(
        monotone_constraints={},
        interactions=[],
        n_jobs=1,
        parameters=default_regression.parameters,
        return_train_score=True,
    )

    model = QuantileCV(
        alpha=0.45,
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=config,
        backend="catboost",
    )
    model.fit(X_train, y_train)

    return model


def test_model_predict_catboost(regression_data, trained_model_catboost):
    """predict should work for the CatBoost backend."""
    _, X_test, _, _ = regression_data
    assert all(np.isreal(trained_model_catboost.predict(X_test)))


def test_model_predict_quantile_catboost(regression_data, trained_quantile_catboost):
    """Quantile predict should work for the CatBoost backend."""
    _, X_test, _, _ = regression_data
    assert all(np.isreal(trained_quantile_catboost.predict(X_test)))


def test_model_score_catboost(regression_data, trained_model_catboost):
    """score should work for the CatBoost backend."""
    _, X_test, _, y_test = regression_data
    assert trained_model_catboost.score(X_test, y_test)


def test_model_explain_catboost(regression_data, trained_model_catboost):
    """explain should work for the CatBoost backend."""
    _, X_test, _, _ = regression_data

    explain = trained_model_catboost.explain(X_test)
    assert explain["shap_values"].shape == (250, 20)


def test_model_performance_catboost(regression_data, trained_model_catboost):
    """CatBoost-backed model should outperform a dummy baseline."""
    X_train, X_test, y_train, y_test = regression_data

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)

    baseline_score = dummy.score(X_test, y_test)
    model_score = trained_model_catboost.score(X_test, y_test)

    assert baseline_score < model_score


def test_regressioncv_forwards_validation_fit_params_to_optimize(
    regression_data, monkeypatch
):
    """fit(**fit_params) should forward X_validation/y_validation into BaseAutoCV.optimize."""
    X_train, _, y_train, _ = regression_data

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0
    )

    from tree_machine.base import BaseAutoCV

    def _spy_optimize(self, *args, **kwargs):
        assert kwargs.get("X_validation") is X_val
        assert kwargs.get("y_validation") is y_val

        class _DummyModel:
            feature_importances_ = np.array([], dtype=float)

        return _DummyModel()

    monkeypatch.setattr(BaseAutoCV, "optimize", _spy_optimize, raising=True)

    model = RegressionCV(
        metric="mse",
        cv=KFold(n_splits=3),
        n_trials=1,
        timeout=1,
        config=default_regression,
    )

    model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)
    assert hasattr(model, "model_")
