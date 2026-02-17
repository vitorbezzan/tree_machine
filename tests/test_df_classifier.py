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
def classification_data():
    """Small classification dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=10,
        n_redundant=0,
        n_classes=3,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


@pytest.fixture(scope="session")
def classifier():
    """Small DFClassifier instance."""
    return DFClassifier(
        metric="cross_entropy",
        n_estimators=2,
        internal_size=8,
        max_depth=2,
        feature_fraction=0.8,
    )


@pytest.fixture(scope="session")
def fitted_classifier(classification_data, classifier):
    """Fitted DFClassifier."""
    X_train, _, y_train, _ = classification_data
    classifier.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    return classifier


@pytest.fixture(scope="session")
def predictions(classification_data, fitted_classifier):
    """Predictions from fitted classifier."""
    _, X_test, _, _ = classification_data
    return fitted_classifier.predict(X_test)


@pytest.fixture(scope="session")
def probabilities(classification_data, fitted_classifier):
    """Probability predictions from fitted classifier."""
    _, X_test, _, _ = classification_data
    return fitted_classifier.predict_proba(X_test)


class TestBeforeFit:
    """Tests for unfitted classifier."""

    def test_predict_raises(self, classification_data):
        """Calling predict before fit should raise."""
        from sklearn.exceptions import NotFittedError

        clf = DFClassifier(
            metric="cross_entropy",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )
        _, X_test, _, _ = classification_data

        with pytest.raises(NotFittedError):
            clf.predict(X_test)


class TestPredictions:
    """Tests for prediction output."""

    def test_shape(self, classification_data, predictions):
        """Predictions are 1D."""
        _, X_test, _, _ = classification_data
        assert predictions.shape == (X_test.shape[0],)

    def test_valid_classes(self, fitted_classifier, predictions):
        """Predictions only contain known classes."""
        assert set(np.unique(predictions)).issubset(set(fitted_classifier.classes_))


class TestProbabilities:
    """Tests for probability predictions."""

    def test_shape(self, classification_data, fitted_classifier, probabilities):
        """Probabilities have correct shape."""
        _, X_test, _, _ = classification_data
        assert probabilities.shape == (X_test.shape[0], len(fitted_classifier.classes_))

    def test_rows_sum_to_one(self, probabilities):
        """Probability rows sum to 1."""
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)

    def test_finite(self, probabilities):
        """Probabilities are finite."""
        assert np.isfinite(probabilities).all()


class TestScore:
    """Tests for score method."""

    def test_returns_finite_float(self, classification_data, fitted_classifier):
        """Score returns a finite float."""
        _, X_test, _, y_test = classification_data
        score = fitted_classifier.score(X_test, y_test)

        assert isinstance(score, (float, np.floating))
        assert np.isfinite(score)


class TestDataFrameHandling:
    """Tests for DataFrame input handling."""

    def test_reorder_columns(self, classification_data, fitted_classifier):
        """Reordered columns don't change probabilities."""
        _, X_test, _, _ = classification_data

        cols_reordered = list(reversed(X_test.columns))
        X_test_reordered = X_test[cols_reordered]

        p1 = fitted_classifier.predict_proba(X_test)
        p2 = fitted_classifier.predict_proba(X_test_reordered)

        np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-7)

    def test_missing_column_raises(self, classification_data, fitted_classifier):
        """Missing columns raise KeyError."""
        _, X_test, _, _ = classification_data
        X_missing = X_test.drop(columns=[X_test.columns[0]])

        with pytest.raises(KeyError):
            fitted_classifier.predict(X_missing)


class TestInputValidation:
    """Tests for input validation."""

    def test_bad_metric_raises(self):
        """Unsupported metric raises ValueError."""
        with pytest.raises(ValueError):
            DFClassifier(
                metric="accuracy",
                n_estimators=2,
                internal_size=8,
                max_depth=2,
                feature_fraction=0.8,
            )

    def test_nan_raises(self, classification_data):
        """NaN values raise ValueError."""
        X_train, _, y_train, _ = classification_data
        X_train = X_train.copy()
        X_train.iloc[0, 0] = np.nan

        clf = DFClassifier(
            metric="cross_entropy",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )

        with pytest.raises(ValueError):
            clf.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

    def test_multilabel_rejected(self, classification_data):
        """Multilabel targets raise ValueError."""
        X_train, _, y_train, _ = classification_data
        y_multi = np.eye(len(np.unique(y_train)))[y_train]

        clf = DFClassifier(
            metric="cross_entropy",
            n_estimators=2,
            internal_size=8,
            max_depth=2,
            feature_fraction=0.8,
        )

        with pytest.raises(ValueError):
            clf.fit(X_train, y_multi, epochs=1, batch_size=32, verbose=0)


class TestCustomLoss:
    """Tests for custom loss function."""

    def test_used_in_score(self, classification_data):
        """Custom loss is used by score."""
        X_train, X_test, y_train, y_test = classification_data

        def ce_loss(y_true, y_pred):
            """Compute categorical cross-entropy loss for testing."""
            return tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            )

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


class TestCompileKwargs:
    """Tests for compile_kwargs passthrough."""

    def test_optimizer_passed(self, classification_data):
        """Compile kwargs are forwarded."""
        X_train, _, y_train, _ = classification_data

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


class TestRegularization:
    """Tests for regularization and dropout."""

    def test_trains_and_predicts(self, classification_data):
        """Model trains with regularization knobs."""
        X_train, X_test, y_train, _ = classification_data

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
