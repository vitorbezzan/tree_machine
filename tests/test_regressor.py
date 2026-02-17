"""Tests for RegressionCV and QuantileCV behavior."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from tree_machine import (
    QuantileCV,
    RegressionCV,
    RegressionCVConfig,
    default_regression,
)


@pytest.fixture(scope="session")
def regression_data():
    """Return a regression train/test split."""
    X, y = make_regression(
        n_samples=1000, n_features=20, n_informative=20, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(20)])
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="session")
def validation_split(regression_data):
    """Provide a validation split for regression data."""
    X_train, _, y_train, _ = regression_data
    return train_test_split(X_train, y_train, test_size=0.2, random_state=0)


@pytest.fixture(scope="session")
def cv():
    """Return a reusable five-fold splitter."""
    return KFold(n_splits=5)


@pytest.fixture(scope="session")
def catboost_config():
    """Return RegressionCV configuration for CatBoost backend."""
    return RegressionCVConfig(
        monotone_constraints={},
        interactions=[],
        n_jobs=1,
        parameters=default_regression.parameters,
        return_train_score=True,
    )


@pytest.fixture(scope="session")
def regression_model(regression_data, cv):
    """Train a RegressionCV model."""
    X_train, _, y_train, _ = regression_data
    return RegressionCV(
        metric="mse", cv=cv, n_trials=50, timeout=120, config=default_regression
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def quantile_model(regression_data, cv):
    """Train a QuantileCV model."""
    X_train, _, y_train, _ = regression_data
    return QuantileCV(
        alpha=0.45, cv=cv, n_trials=50, timeout=120, config=default_regression
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def catboost_model(regression_data, cv, catboost_config):
    """Train a CatBoost-backed RegressionCV model."""
    X_train, _, y_train, _ = regression_data
    return RegressionCV(
        metric="mse",
        cv=cv,
        n_trials=50,
        timeout=120,
        config=catboost_config,
        backend="catboost",
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def catboost_quantile(regression_data, cv, catboost_config):
    """Train a CatBoost-backed QuantileCV model."""
    X_train, _, y_train, _ = regression_data
    return QuantileCV(
        alpha=0.45,
        cv=cv,
        n_trials=50,
        timeout=120,
        config=catboost_config,
        backend="catboost",
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def dummy_baseline(regression_data):
    """Fit a dummy regressor baseline."""
    X_train, _, y_train, _ = regression_data
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    return dummy


@pytest.fixture(scope="session")
def predictions(regression_data, regression_model):
    """Return regression predictions for the held-out set."""
    _, X_test, _, _ = regression_data
    return regression_model.predict(X_test)


@pytest.fixture(scope="session")
def explanation(regression_data, regression_model):
    """Explain the regression model predictions."""
    _, X_test, _, _ = regression_data
    return regression_model.explain(X_test)


@pytest.fixture(scope="session")
def catboost_predictions(regression_data, catboost_model):
    """Return predictions from the CatBoost regression model."""
    _, X_test, _, _ = regression_data
    return catboost_model.predict(X_test)


@pytest.fixture(scope="session")
def catboost_explanation(regression_data, catboost_model):
    """Explain the CatBoost regression model predictions."""
    _, X_test, _, _ = regression_data
    return catboost_model.explain(X_test)


class TestRegression:
    """Standard regression model checks."""

    def test_predict_returns_real(self, predictions):
        """Predictions should be real-valued."""
        assert all(np.isreal(predictions))

    def test_score(self, regression_data, regression_model):
        """Model should return a finite score."""
        _, X_test, _, y_test = regression_data
        assert regression_model.score(X_test, y_test)

    def test_beats_baseline(self, regression_data, regression_model, dummy_baseline):
        """Model score should beat the dummy baseline."""
        _, X_test, _, y_test = regression_data
        assert regression_model.score(X_test, y_test) > dummy_baseline.score(
            X_test, y_test
        )

    def test_explain_shape(self, explanation):
        """Explanation output should have expected shape."""
        assert explanation["shap_values"].shape == (250, 20)


class TestQuantile:
    """Quantile regression checks."""

    def test_predict_returns_real(self, regression_data, quantile_model):
        """Quantile predictions should be real-valued."""
        _, X_test, _, _ = regression_data
        assert all(np.isreal(quantile_model.predict(X_test)))


class TestCatBoostRegression:
    """CatBoost regression model checks."""

    def test_predict_returns_real(self, catboost_predictions):
        """CatBoost predictions should be real-valued."""
        assert all(np.isreal(catboost_predictions))

    def test_score(self, regression_data, catboost_model):
        """CatBoost model should return a finite score."""
        _, X_test, _, y_test = regression_data
        assert catboost_model.score(X_test, y_test)

    def test_beats_baseline(self, regression_data, catboost_model, dummy_baseline):
        """CatBoost model should beat the dummy baseline."""
        _, X_test, _, y_test = regression_data
        assert catboost_model.score(X_test, y_test) > dummy_baseline.score(
            X_test, y_test
        )

    def test_explain_shape(self, catboost_explanation):
        """CatBoost explanation output should have expected shape."""
        assert catboost_explanation["shap_values"].shape == (250, 20)


class TestCatBoostQuantile:
    """CatBoost quantile regression checks."""

    def test_predict_returns_real(self, regression_data, catboost_quantile):
        """CatBoost quantile predictions should be real-valued."""
        _, X_test, _, _ = regression_data
        assert all(np.isreal(catboost_quantile.predict(X_test)))


class TestValidationFit:
    """Validation passthrough and fitting behavior."""

    def test_forwards_validation_to_optimize(self, validation_split, monkeypatch):
        """Ensure validation data is forwarded to optimize."""
        X_tr, X_val, y_tr, y_val = validation_split

        from tree_machine.base import BaseAutoCV

        captured = {}

        def spy_optimize(self, *args, **kwargs):
            """Capture validation arguments passed to optimize."""
            captured["X_val"] = kwargs.get("X_validation")
            captured["y_val"] = kwargs.get("y_validation")

            class DummyModel:
                """Minimal dummy model placeholder."""

                feature_importances_ = np.array([], dtype=float)

            return DummyModel()

        monkeypatch.setattr(BaseAutoCV, "optimize", spy_optimize)

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert captured["X_val"] is X_val
        assert captured["y_val"] is y_val

    def test_model_fitted_after_validation(self, validation_split, monkeypatch):
        """Ensure model_ attribute is set after validation fit."""
        X_tr, X_val, y_tr, y_val = validation_split

        from tree_machine.base import BaseAutoCV

        def spy_optimize(self, *args, **kwargs):
            """Return a dummy model during optimize."""

            class DummyModel:
                """Minimal dummy model placeholder."""

                feature_importances_ = np.array([], dtype=float)

            return DummyModel()

        monkeypatch.setattr(BaseAutoCV, "optimize", spy_optimize)

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert hasattr(model, "model_")


class TestValidationObjective:
    """Validation objective behavior for RegressionCV."""

    def test_scorer_called(self, validation_split, monkeypatch):
        """Scorer should be called during validation fit."""
        X_tr, X_val, y_tr, y_val = validation_split
        calls = {"count": 0}

        def spy_scorer(estimator, X, y):
            """Count scorer invocations and return a score."""
            calls["count"] += 1
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)

        monkeypatch.setattr(RegressionCV, "scorer", property(lambda self: spy_scorer))

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=2,
            timeout=30,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert calls["count"] >= 1

    def test_validation_shape(self, validation_split, monkeypatch):
        """Validation data passed to scorer should match expected shapes."""
        X_tr, X_val, y_tr, y_val = validation_split
        calls = {}

        def spy_scorer(estimator, X, y):
            """Capture the shapes of validation data passed to scorer."""
            calls["X"] = X
            calls["y"] = y
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)

        monkeypatch.setattr(RegressionCV, "scorer", property(lambda self: spy_scorer))

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=2,
            timeout=30,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert np.asarray(calls["X"]).shape == (X_val.shape[0], X_val.shape[1])
        assert np.asarray(calls["y"]).shape == (y_val.shape[0],)

    def test_cv_results_shape(self, validation_split, monkeypatch):
        """cv_results should have a finite score entry."""
        X_tr, X_val, y_tr, y_val = validation_split

        def spy_scorer(estimator, X, y):
            """Return a finite score for validation data."""
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)

        monkeypatch.setattr(RegressionCV, "scorer", property(lambda self: spy_scorer))

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=2,
            timeout=30,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert model.cv_results.shape == (1,)
        assert np.isfinite(model.cv_results[0])

    def test_fitted_attributes(self, validation_split, monkeypatch):
        """Fitted model should expose fitted attributes after validation."""
        X_tr, X_val, y_tr, y_val = validation_split

        def spy_scorer(estimator, X, y):
            """Return a finite score for validation data."""
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)

        monkeypatch.setattr(RegressionCV, "scorer", property(lambda self: spy_scorer))

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=2,
            timeout=30,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert hasattr(model, "model_")
        assert hasattr(model, "study_")
        assert hasattr(model, "best_params_")
        assert hasattr(model, "feature_importances_")

    def test_best_params_not_empty(self, validation_split, monkeypatch):
        """best_params_ should contain entries after optimization."""
        X_tr, X_val, y_tr, y_val = validation_split

        def spy_scorer(estimator, X, y):
            """Return a finite score for validation data."""
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)

        monkeypatch.setattr(RegressionCV, "scorer", property(lambda self: spy_scorer))

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=2,
            timeout=30,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert len(model.best_params_) > 0

    def test_feature_importances_type(self, validation_split, monkeypatch):
        """feature_importances_ should be a numpy array."""
        X_tr, X_val, y_tr, y_val = validation_split

        def spy_scorer(estimator, X, y):
            """Return a finite score for validation data."""
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)

        monkeypatch.setattr(RegressionCV, "scorer", property(lambda self: spy_scorer))

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=2,
            timeout=30,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert isinstance(model.feature_importances_, np.ndarray)

    def test_predict_shape(self, validation_split, monkeypatch):
        """Predictions should match the validation feature row count."""
        X_tr, X_val, y_tr, y_val = validation_split

        def spy_scorer(estimator, X, y):
            """Return a finite score for validation data."""
            y_pred = estimator.predict(X)
            return -mean_squared_error(y, y_pred)

        monkeypatch.setattr(RegressionCV, "scorer", property(lambda self: spy_scorer))

        model = RegressionCV(
            metric="mse",
            cv=KFold(n_splits=3),
            n_trials=2,
            timeout=30,
            config=default_regression,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert model.predict(X_val).shape[0] == X_val.shape[0]
