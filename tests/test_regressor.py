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


def test_regressioncv_uses_validation_set_for_optimization(regression_data):
    """When validation data is provided, optimization should use validation scoring, not CV."""
    X_train, _, y_train, _ = regression_data

    # Create explicit train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Fit model with validation data
    model = RegressionCV(
        metric="mse",
        cv=KFold(n_splits=3),
        n_trials=3,  # Use multiple trials to ensure optimization works
        timeout=10,
        config=default_regression,
    )
    model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

    # Verify model is fitted with expected attributes
    assert hasattr(model, "model_")
    assert hasattr(model, "study_")
    assert hasattr(model, "best_params_")
    assert hasattr(model, "feature_importances_")

    # Verify cv_results contains a single validation score (not multiple CV folds)
    # When using validation set, cv_results should have test_score as array with 1 element
    cv_results = model.cv_results
    assert isinstance(cv_results, np.ndarray)
    assert len(cv_results) == 1, (
        "Validation path should produce single score, "
        f"but got {len(cv_results)} scores"
    )
    # Verify the score is a finite number
    assert np.isfinite(cv_results[0]), "Validation score should be finite"

    # Verify the model can make predictions
    predictions = model.predict(X_val)
    assert len(predictions) == len(y_val)
    assert all(np.isfinite(predictions))

    # Verify the score is reasonable (model should learn something)
    score = model.score(X_val, y_val)
    assert score > -1.0  # R² should be > -1 (can be negative if model is worse than mean)


def test_regressioncv_validation_requires_both_X_and_y(regression_data):
    """Providing only X_validation or y_validation should raise an error with clear message."""
    X_train, _, y_train, _ = regression_data

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model = RegressionCV(
        metric="mse",
        cv=KFold(n_splits=3),
        n_trials=1,
        timeout=1,
        config=default_regression,
    )

    # Test providing only X_validation
    with pytest.raises(
        ValueError, match="Got X_validation but y_validation is missing"
    ):
        model.fit(X_tr, y_tr, X_validation=X_val)

    # Test providing only y_validation
    with pytest.raises(
        ValueError, match="Got y_validation but X_validation is missing"
    ):
        model.fit(X_tr, y_tr, y_validation=y_val)
