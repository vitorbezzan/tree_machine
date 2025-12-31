"""Tests for the experimental deep-forest classifier (DFClassifier)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

tf = pytest.importorskip("tensorflow")

from tree_machine.deep_trees.classifier import DFClassifier  # noqa


@pytest.fixture(scope="session")
def classification_data_small_df():
    """Return a small train/test classification split as pandas DataFrames."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=10,
        n_redundant=0,
        n_classes=3,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture()
def classifier():
    """Create a small DFClassifier instance suitable for fast unit tests."""
    return DFClassifier(
        metric="cross_entropy",
        n_estimators=2,
        internal_size=8,
        max_depth=2,
        feature_fraction=0.8,
    )


def test_predict_before_fit_raises(classification_data_small_df, classifier) -> None:
    """Calling predict before fit should raise."""
    _, X_test, _, _ = classification_data_small_df

    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        classifier.predict(X_test)


def test_fit_predict_output_is_1d_and_valid_classes(
    classification_data_small_df, classifier
) -> None:
    """After fitting, predictions should be 1D and only contain known classes."""
    X_train, X_test, y_train, _ = classification_data_small_df

    classifier.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    preds = classifier.predict(X_test)

    assert preds.shape == (X_test.shape[0],)
    assert set(np.unique(preds)).issubset(set(classifier.classes_))


def test_predict_proba_shape_and_rows_sum_to_one(
    classification_data_small_df, classifier
) -> None:
    """predict_proba should return (n_samples, n_classes) and rows sum to 1."""
    X_train, X_test, y_train, _ = classification_data_small_df

    classifier.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    proba = classifier.predict_proba(X_test)

    assert proba.shape == (X_test.shape[0], len(classifier.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    assert np.isfinite(proba).all()


def test_score_returns_finite_float(classification_data_small_df, classifier) -> None:
    """score() should return a finite scalar float."""
    X_train, X_test, y_train, y_test = classification_data_small_df

    classifier.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    score = classifier.score(X_test, y_test)

    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_dataframe_column_reorder_is_handled(
    classification_data_small_df, classifier
) -> None:
    """Reordered DataFrame columns should not change predicted probabilities."""
    X_train, X_test, y_train, _ = classification_data_small_df

    classifier.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    cols_reordered = list(reversed(X_test.columns))
    X_test_reordered = X_test[cols_reordered]

    p1 = classifier.predict_proba(X_test)
    p2 = classifier.predict_proba(X_test_reordered)

    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-7)


def test_missing_dataframe_column_raises(
    classification_data_small_df, classifier
) -> None:
    """Missing required feature columns should raise KeyError."""
    X_train, X_test, y_train, _ = classification_data_small_df

    classifier.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

    X_missing = X_test.drop(columns=[X_test.columns[0]])

    with pytest.raises(KeyError):
        classifier.predict(X_missing)


def test_bad_metric_raises() -> None:
    """Unsupported built-in metric keys should be rejected."""
    with pytest.raises(ValueError):
        DFClassifier(
            metric="accuracy",  # not supported as metric key
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )


def test_nan_in_X_raises(classification_data_small_df, classifier) -> None:
    """NaN values in X should be rejected by input validation."""
    X_train, _, y_train, _ = classification_data_small_df

    X_train = X_train.copy()
    X_train.iloc[0, 0] = np.nan

    with pytest.raises(ValueError):
        classifier.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)


def test_multilabel_y_rejected(classification_data_small_df, classifier) -> None:
    """Multilabel indicator targets should be rejected."""
    X_train, _, y_train, _ = classification_data_small_df

    y_multi = np.eye(len(np.unique(y_train)))[y_train]

    with pytest.raises(ValueError):
        classifier.fit(X_train, y_multi, epochs=1, batch_size=32, verbose=0)


def test_custom_loss_callable_is_used_in_score(classification_data_small_df) -> None:
    """A custom loss callable should be used by score() when provided."""
    X_train, X_test, y_train, y_test = classification_data_small_df

    def ce_loss(y_true, y_pred):
        import tensorflow as tf

        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

    clf = DFClassifier(
        metric="cross_entropy",
        n_estimators=2,
        internal_size=8,
        max_depth=2,
        feature_fraction=0.8,
        loss=ce_loss,
    )

    clf.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

    score = clf.score(X_test, y_test)
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_compile_kwargs_are_passed_through(classification_data_small_df) -> None:
    """compile_kwargs should be forwarded to the underlying Keras Model.compile."""
    X_train, _, y_train, _ = classification_data_small_df

    clf = DFClassifier(
        metric="cross_entropy",
        n_estimators=2,
        internal_size=8,
        max_depth=2,
        feature_fraction=0.8,
        compile_kwargs={"optimizer": "sgd"},
    )

    clf.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

    opt_name = getattr(clf.model_.optimizer, "name", str(clf.model_.optimizer))
    assert "sgd" in str(opt_name).lower()


def test_regularization_and_dropout_knobs_train_and_predict(
    classification_data_small_df,
) -> None:
    """The forest should accept regularization/dropout knobs and still train."""
    X_train, X_test, y_train, _ = classification_data_small_df

    clf = DFClassifier(
        metric="cross_entropy",
        n_estimators=2,
        internal_size=8,
        max_depth=2,
        feature_fraction=0.8,
        decision_l2=1e-3,
        leaf_l2=1e-3,
        feature_dropout=0.1,
        routing_dropout=0.1,
    )

    clf.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

    p1 = clf.predict_proba(X_test)
    p2 = clf.predict_proba(X_test)

    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-7)
    assert np.isfinite(p1).all()
