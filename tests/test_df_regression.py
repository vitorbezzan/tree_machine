"""Tests for the experimental deep-forest regressor (DFRegression)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

tf = pytest.importorskip("tensorflow")

from tree_machine.deep_trees.regression import DFRegression  # noqa


@pytest.fixture(scope="session")
def regression_data_small_df():
    """Return a small train/test regression split as pandas DataFrames."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=10,
        noise=0.1,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture()
def regressor():
    """Create a small DFRegression instance suitable for fast unit tests."""
    return DFRegression(
        metric="mse",
        n_estimators=2,
        internal_size=8,
        max_depth=2,
        feature_fraction=0.8,
    )


def test_predict_before_fit_raises(regression_data_small_df, regressor) -> None:
    """Calling predict before fit should raise."""
    _, X_test, _, _ = regression_data_small_df
    with pytest.raises((ValueError, RuntimeError)):
        regressor.predict(X_test)


def test_fit_predict_output_is_1d_and_finite(
    regression_data_small_df, regressor
) -> None:
    """After fitting, predictions should be 1D and finite."""
    X_train, X_test, y_train, _ = regression_data_small_df

    regressor.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    preds = regressor.predict(X_test)

    assert preds.shape == (X_test.shape[0],)
    assert np.isfinite(preds).all()


def test_score_returns_finite_float(regression_data_small_df, regressor) -> None:
    """score() should return a finite scalar float."""
    X_train, X_test, y_train, y_test = regression_data_small_df

    regressor.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    score = regressor.score(X_test, y_test)

    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_dataframe_column_reorder_is_handled(
    regression_data_small_df, regressor
) -> None:
    """Reordered DataFrame columns should not change predictions."""
    X_train, X_test, y_train, _ = regression_data_small_df

    regressor.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    cols_reordered = list(reversed(X_test.columns))
    X_test_reordered = X_test[cols_reordered]

    p1 = regressor.predict(X_test)
    p2 = regressor.predict(X_test_reordered)

    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-7)


def test_missing_dataframe_column_raises(regression_data_small_df, regressor) -> None:
    """Missing required feature columns should raise KeyError."""
    X_train, X_test, y_train, _ = regression_data_small_df

    regressor.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

    X_missing = X_test.drop(columns=[X_test.columns[0]])

    with pytest.raises(KeyError):
        regressor.predict(X_missing)


def test_bad_metric_raises() -> None:
    """Unsupported built-in metric keys should be rejected."""
    with pytest.raises(ValueError):
        DFRegression(
            metric="rmse",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )


def test_nan_in_X_raises(regression_data_small_df, regressor) -> None:
    """NaN values in X should be rejected by input validation."""
    X_train, X_test, y_train, _ = regression_data_small_df

    X_train = X_train.copy()
    X_train.iloc[0, 0] = np.nan

    with pytest.raises(ValueError):
        regressor.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)


def test_multioutput_y_rejected(regression_data_small_df, regressor) -> None:
    """Multi-output regression targets should be rejected."""
    X_train, _, y_train, _ = regression_data_small_df

    y_multi = np.column_stack([y_train, y_train])

    with pytest.raises(ValueError):
        regressor.fit(X_train, y_multi, epochs=1, batch_size=32, verbose=0)


def test_custom_loss_callable_is_used_in_score(regression_data_small_df) -> None:
    """A custom loss callable should be used by score() when provided."""
    X_train, X_test, y_train, y_test = regression_data_small_df

    def l1_loss(y_true, y_pred):
        import tensorflow as tf

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

    y_true = np.asarray(y_test).reshape(-1, 1)
    y_pred = np.asarray(reg.predict(X_test)).reshape(-1, 1)
    expected = -np.mean(np.abs(y_true - y_pred))

    np.testing.assert_allclose(score, expected, rtol=0, atol=1e-6)


def test_compile_kwargs_are_passed_through(regression_data_small_df) -> None:
    """compile_kwargs should be forwarded to the underlying Keras Model.compile."""
    X_train, _, y_train, _ = regression_data_small_df

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


def test_regularization_and_dropout_knobs_train_and_predict(
    regression_data_small_df,
) -> None:
    """The forest should accept regularization/dropout knobs and still train.

    Dropout must be inactive at inference (predict) time.
    """
    X_train, X_test, y_train, _ = regression_data_small_df

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
