"""Tests for the experimental deep-forest regressor (DFRegression)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

tf = pytest.importorskip("tensorflow")

from tree_machine.deep_trees.regression import DFRegression  # noqa


@pytest.fixture(scope="session")
def regression_data():
    """Small regression dataset."""
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=10, noise=0.1, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="session")
def regressor():
    """Small DFRegression instance."""
    return DFRegression(
        metric="mse", n_estimators=2, internal_size=8, max_depth=2, feature_fraction=0.8
    )


@pytest.fixture(scope="session")
def fitted_regressor(regression_data, regressor):
    """Fitted DFRegression."""
    X_train, _, y_train, _ = regression_data
    regressor.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    return regressor


@pytest.fixture(scope="session")
def predictions(regression_data, fitted_regressor):
    """Predictions from fitted regressor."""
    _, X_test, _, _ = regression_data
    return fitted_regressor.predict(X_test)


class TestBeforeFit:
    """Tests for unfitted regressor."""

    def test_predict_raises(self, regression_data):
        """Calling predict before fit should raise."""
        reg = DFRegression(
            metric="mse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )
        _, X_test, _, _ = regression_data

        with pytest.raises(NotFittedError):
            reg.predict(X_test)


class TestPredictions:
    """Tests for prediction output."""

    def test_shape(self, regression_data, predictions):
        """Predictions are 1D."""
        _, X_test, _, _ = regression_data
        assert predictions.shape == (X_test.shape[0],)

    def test_finite(self, predictions):
        """Predictions are finite."""
        assert np.isfinite(predictions).all()


class TestScore:
    """Tests for score method."""

    def test_returns_finite_float(self, regression_data, fitted_regressor):
        """Score returns a finite float."""
        _, X_test, _, y_test = regression_data
        score = fitted_regressor.score(X_test, y_test)

        assert isinstance(score, (float, np.floating))
        assert np.isfinite(score)


class TestDataFrameHandling:
    """Tests for DataFrame input handling."""

    def test_reorder_columns(self, regression_data, fitted_regressor):
        """Reordered columns don't change predictions."""
        _, X_test, _, _ = regression_data

        cols_reordered = list(reversed(X_test.columns))
        X_test_reordered = X_test[cols_reordered]

        p1 = fitted_regressor.predict(X_test)
        p2 = fitted_regressor.predict(X_test_reordered)

        np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-7)

    def test_missing_column_raises(self, regression_data, fitted_regressor):
        """Missing columns raise KeyError."""
        _, X_test, _, _ = regression_data
        X_missing = X_test.drop(columns=[X_test.columns[0]])

        with pytest.raises(KeyError):
            fitted_regressor.predict(X_missing)


class TestInputValidation:
    """Tests for input validation."""

    def test_bad_metric_raises(self):
        """Unsupported metric raises ValueError."""
        with pytest.raises(ValueError):
            DFRegression(
                metric="rmse",
                n_estimators=2,
                internal_size=8,
                max_depth=2,
                feature_fraction=0.8,
            )

    def test_nan_raises(self, regression_data):
        """NaN values raise ValueError."""
        X_train, _, y_train, _ = regression_data
        X_train = X_train.copy()
        X_train.iloc[0, 0] = np.nan

        reg = DFRegression(
            metric="mse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )

        with pytest.raises(ValueError):
            reg.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

    def test_multioutput_rejected(self, regression_data):
        """Multi-output targets raise ValueError."""
        X_train, _, y_train, _ = regression_data
        y_multi = np.column_stack([y_train, y_train])

        reg = DFRegression(
            metric="mse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )

        with pytest.raises(ValueError):
            reg.fit(X_train, y_multi, epochs=1, batch_size=32, verbose=0)


class TestCustomLoss:
    """Tests for custom loss function."""

    def test_used_in_score(self, regression_data):
        """Custom loss is used by score."""
        X_train, X_test, y_train, y_test = regression_data

        def l1_loss(y_true, y_pred):
            """Compute mean absolute error loss for testing."""
            return tf.reduce_mean(tf.abs(y_true - y_pred))

        reg = DFRegression(
            metric="mse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
            loss=l1_loss,
        )
        reg.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

        score = reg.score(X_test, y_test)
        assert isinstance(score, (float, np.floating))
        assert np.isfinite(score)

    def test_score_matches_expected(self, regression_data):
        """Custom loss score matches manual calculation."""
        X_train, X_test, y_train, y_test = regression_data

        def l1_loss(y_true, y_pred):
            """Compute mean absolute error loss for testing."""
            return tf.reduce_mean(tf.abs(y_true - y_pred))

        reg = DFRegression(
            metric="mse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
            loss=l1_loss,
        )
        reg.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

        y_true = np.asarray(y_test).reshape(-1, 1)
        y_pred = np.asarray(reg.predict(X_test)).reshape(-1, 1)
        expected = -np.mean(np.abs(y_true - y_pred))

        np.testing.assert_allclose(
            reg.score(X_test, y_test), expected, rtol=0, atol=1e-6
        )


class TestCompileKwargs:
    """Tests for compile_kwargs passthrough."""

    def test_optimizer_passed(self, regression_data):
        """Compile kwargs are forwarded."""
        X_train, _, y_train, _ = regression_data

        reg = DFRegression(
            metric="mse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
            compile_kwargs={"optimizer": "sgd"},
        )
        reg.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

        opt_name = getattr(reg.model_.optimizer, "name", str(reg.model_.optimizer))
        assert "sgd" in str(opt_name).lower()


class TestRegularization:
    """Tests for regularization and dropout."""

    def test_trains_and_predicts(self, regression_data):
        """Model trains with regularization knobs."""
        X_train, X_test, y_train, _ = regression_data

        reg = DFRegression(
            metric="mse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
            decision_l2=1e-3,
            leaf_l2=1e-3,
            feature_dropout=0.1,
            routing_dropout=0.1,
        )
        reg.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

        p1 = reg.predict(X_test)
        p2 = reg.predict(X_test)

        assert p1.shape == (X_test.shape[0],)
        np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-7)
        assert np.isfinite(p1).all()
