"""Tests for pickle conformance with custom metric functions."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from tree_machine import (
    ClassifierCV,
    RegressionCV,
    default_classifier,
    default_regression,
)


def custom_accuracy(y_true, y_pred):
    """Compute accuracy using sklearn."""
    return accuracy_score(y_true, y_pred)


def custom_f1(y_true, y_pred):
    """Compute weighted F1 using sklearn."""
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def custom_r2(y_true, y_pred):
    """Compute R2 using sklearn."""
    return r2_score(y_true, y_pred)


def custom_mse(y_true, y_pred):
    """Return negative MSE so higher values are better."""
    return -mean_squared_error(y_true, y_pred)


def custom_mae(y_true, y_pred):
    """Return negative MAE so higher values are better."""
    return -mean_absolute_error(y_true, y_pred)


@pytest.fixture(scope="session")
def regression_data():
    """Regression dataset."""
    X, y = make_regression(
        n_samples=500, n_features=15, n_informative=10, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(15)])
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="session")
def classification_data():
    """Classification dataset."""
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=15, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(20)])
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="session")
def stratified_cv():
    """Stratified cross-validation splitter."""
    return StratifiedKFold(n_splits=3)


@pytest.fixture(scope="session")
def kfold_cv():
    """K-fold cross-validation splitter."""
    return KFold(n_splits=3)


@pytest.fixture(scope="session")
def classifier_accuracy(classification_data, stratified_cv):
    """Classifier with custom accuracy metric."""
    X_train, _, y_train, _ = classification_data
    return ClassifierCV(
        metric=custom_accuracy,
        cv=stratified_cv,
        n_trials=5,
        timeout=60,
        config=default_classifier,
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def classifier_f1(classification_data, stratified_cv):
    """Classifier with custom F1 metric."""
    X_train, _, y_train, _ = classification_data
    return ClassifierCV(
        metric=custom_f1,
        cv=stratified_cv,
        n_trials=5,
        timeout=60,
        config=default_classifier,
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def regressor_r2(regression_data, kfold_cv):
    """Regressor with custom R2 metric."""
    X_train, _, y_train, _ = regression_data
    return RegressionCV(
        metric=custom_r2, cv=kfold_cv, n_trials=5, timeout=60, config=default_regression
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def regressor_mse(regression_data, kfold_cv):
    """Regressor with custom MSE metric."""
    X_train, _, y_train, _ = regression_data
    return RegressionCV(
        metric=custom_mse,
        cv=kfold_cv,
        n_trials=5,
        timeout=60,
        config=default_regression,
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def pickled_classifier_accuracy(classifier_accuracy):
    """Pickled and restored classifier with accuracy metric."""
    return pickle.loads(pickle.dumps(classifier_accuracy))


@pytest.fixture(scope="session")
def pickled_classifier_f1(classifier_f1):
    """Pickled and restored classifier with F1 metric."""
    return pickle.loads(pickle.dumps(classifier_f1))


@pytest.fixture(scope="session")
def pickled_regressor_r2(regressor_r2):
    """Pickled and restored regressor with R2 metric."""
    return pickle.loads(pickle.dumps(regressor_r2))


@pytest.fixture(scope="session")
def pickled_regressor_mse(regressor_mse):
    """Pickled and restored regressor with MSE metric."""
    return pickle.loads(pickle.dumps(regressor_mse))


class TestClassifierPickleBytes:
    """Tests for classifier pickle to bytes."""

    def test_has_model(self, pickled_classifier_accuracy):
        """Restored model has model_ attribute."""
        assert hasattr(pickled_classifier_accuracy, "model_")

    def test_has_metric(self, pickled_classifier_accuracy):
        """Restored model has metric attribute."""
        assert hasattr(pickled_classifier_accuracy, "metric")

    def test_metric_callable(self, pickled_classifier_accuracy):
        """Restored metric is callable."""
        assert callable(pickled_classifier_accuracy.metric)


class TestClassifierPickleFile:
    """Tests for classifier pickle to file."""

    def test_roundtrip_has_model(self, classifier_f1):
        """File roundtrip preserves model_ attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_file = Path(tmpdir) / "classifier.pkl"

            with open(pickle_file, "wb") as f:
                pickle.dump(classifier_f1, f)

            with open(pickle_file, "rb") as f:
                restored = pickle.load(f)

            assert hasattr(restored, "model_")
            assert callable(restored.metric)


class TestClassifierPredictions:
    """Tests for classifier prediction consistency."""

    def test_predictions_consistent(
        self, classification_data, classifier_accuracy, pickled_classifier_accuracy
    ):
        """Predictions are consistent after pickle."""
        _, X_test, _, _ = classification_data
        before = classifier_accuracy.predict(X_test)
        after = pickled_classifier_accuracy.predict(X_test)
        np.testing.assert_array_equal(before, after)

    def test_proba_consistent(
        self, classification_data, classifier_f1, pickled_classifier_f1
    ):
        """Probabilities are consistent after pickle."""
        _, X_test, _, _ = classification_data
        before = classifier_f1.predict_proba(X_test)
        after = pickled_classifier_f1.predict_proba(X_test)
        np.testing.assert_array_almost_equal(before, after)


class TestClassifierScore:
    """Tests for classifier score consistency."""

    def test_score_consistent(
        self, classification_data, classifier_accuracy, pickled_classifier_accuracy
    ):
        """Score is consistent after pickle."""
        _, X_test, _, y_test = classification_data
        before = classifier_accuracy.score(X_test, y_test)
        after = pickled_classifier_accuracy.score(X_test, y_test)
        assert before == after


class TestClassifierMetric:
    """Tests for classifier metric functionality."""

    def test_metric_callable_with_arrays(self, pickled_classifier_accuracy):
        """Metric works with array inputs."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        result = pickled_classifier_accuracy.metric(y_true, y_pred)
        assert isinstance(result, (int, float, np.number))


class TestClassifierFeatureImportances:
    """Tests for classifier feature importances."""

    def test_preserved(self, classifier_accuracy, pickled_classifier_accuracy):
        """Feature importances are preserved."""
        np.testing.assert_array_equal(
            classifier_accuracy.feature_importances_,
            pickled_classifier_accuracy.feature_importances_,
        )


class TestClassifierMultipleCycles:
    """Tests for multiple pickle cycles."""

    def test_predictions_stable(self, classification_data, classifier_f1):
        """Predictions stable through multiple pickle cycles."""
        _, X_test, _, _ = classification_data
        original = classifier_f1.predict(X_test)
        current = classifier_f1

        for _ in range(3):
            current = pickle.loads(pickle.dumps(current))

        final = current.predict(X_test)
        np.testing.assert_array_equal(original, final)


class TestRegressorPickleBytes:
    """Tests for regressor pickle to bytes."""

    def test_has_model(self, pickled_regressor_r2):
        """Restored model has model_ attribute."""
        assert hasattr(pickled_regressor_r2, "model_")

    def test_has_metric(self, pickled_regressor_r2):
        """Restored model has metric attribute."""
        assert hasattr(pickled_regressor_r2, "metric")

    def test_metric_callable(self, pickled_regressor_r2):
        """Restored metric is callable."""
        assert callable(pickled_regressor_r2.metric)


class TestRegressorPickleFile:
    """Tests for regressor pickle to file."""

    def test_roundtrip_has_model(self, regressor_mse):
        """File roundtrip preserves model_ attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_file = Path(tmpdir) / "regressor.pkl"

            with open(pickle_file, "wb") as f:
                pickle.dump(regressor_mse, f)

            with open(pickle_file, "rb") as f:
                restored = pickle.load(f)

            assert hasattr(restored, "model_")
            assert callable(restored.metric)


class TestRegressorPredictions:
    """Tests for regressor prediction consistency."""

    def test_predictions_consistent(
        self, regression_data, regressor_r2, pickled_regressor_r2
    ):
        """Predictions are consistent after pickle."""
        _, X_test, _, _ = regression_data
        before = regressor_r2.predict(X_test)
        after = pickled_regressor_r2.predict(X_test)
        np.testing.assert_array_almost_equal(before, after)


class TestRegressorScore:
    """Tests for regressor score consistency."""

    def test_score_r2_consistent(
        self, regression_data, regressor_r2, pickled_regressor_r2
    ):
        """R2 score is consistent after pickle."""
        _, X_test, _, y_test = regression_data
        before = regressor_r2.score(X_test, y_test)
        after = pickled_regressor_r2.score(X_test, y_test)
        assert abs(before - after) < 1e-10

    def test_score_mse_consistent(
        self, regression_data, regressor_mse, pickled_regressor_mse
    ):
        """MSE score is consistent after pickle."""
        _, X_test, _, y_test = regression_data
        before = regressor_mse.score(X_test, y_test)
        after = pickled_regressor_mse.score(X_test, y_test)
        assert abs(before - after) < 1e-10


class TestRegressorMetric:
    """Tests for regressor metric functionality."""

    def test_metric_callable_with_arrays(self, pickled_regressor_r2):
        """Metric works with array inputs."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        result = pickled_regressor_r2.metric(y_true, y_pred)
        assert isinstance(result, (int, float, np.number))


class TestRegressorFeatureImportances:
    """Tests for regressor feature importances."""

    def test_preserved(self, regressor_r2, pickled_regressor_r2):
        """Feature importances are preserved."""
        np.testing.assert_array_equal(
            regressor_r2.feature_importances_, pickled_regressor_r2.feature_importances_
        )


class TestRegressorMultipleCycles:
    """Tests for multiple pickle cycles."""

    def test_predictions_stable(self, regression_data, regressor_mse):
        """Predictions stable through multiple pickle cycles."""
        _, X_test, _, _ = regression_data
        original = regressor_mse.predict(X_test)
        current = regressor_mse

        for _ in range(3):
            current = pickle.loads(pickle.dumps(current))

        final = current.predict(X_test)
        np.testing.assert_array_almost_equal(original, final)


class TestClassifierCV:
    """Tests for classifier CV attribute preservation."""

    def test_cv_preserved(self, classification_data, stratified_cv):
        """CV attribute is preserved."""
        X_train, _, y_train, _ = classification_data
        model = ClassifierCV(
            metric=custom_accuracy,
            cv=stratified_cv,
            n_trials=5,
            timeout=60,
            config=default_classifier,
        ).fit(X_train, y_train)

        restored = pickle.loads(pickle.dumps(model))

        assert restored.cv is not None
        assert isinstance(restored.cv, StratifiedKFold)
        assert restored.cv.n_splits == 3


class TestRegressorCV:
    """Tests for regressor CV attribute preservation."""

    def test_cv_preserved(self, regression_data, kfold_cv):
        """CV attribute is preserved."""
        X_train, _, y_train, _ = regression_data
        model = RegressionCV(
            metric=custom_r2,
            cv=kfold_cv,
            n_trials=5,
            timeout=60,
            config=default_regression,
        ).fit(X_train, y_train)

        restored = pickle.loads(pickle.dumps(model))

        assert restored.cv is not None
        assert isinstance(restored.cv, KFold)
        assert restored.cv.n_splits == 3


class TestClassifierFeatureNames:
    """Tests for classifier feature names preservation."""

    def test_preserved(self, classification_data, stratified_cv):
        """Feature names are preserved."""
        X_train, _, y_train, _ = classification_data
        model = ClassifierCV(
            metric=custom_f1,
            cv=stratified_cv,
            n_trials=5,
            timeout=60,
            config=default_classifier,
        ).fit(X_train, y_train)

        before = model.feature_names_.copy()
        restored = pickle.loads(pickle.dumps(model))

        assert restored.feature_names_ == before


class TestRegressorFeatureNames:
    """Tests for regressor feature names preservation."""

    def test_preserved(self, regression_data, kfold_cv):
        """Feature names are preserved."""
        X_train, _, y_train, _ = regression_data
        model = RegressionCV(
            metric=custom_mse,
            cv=kfold_cv,
            n_trials=5,
            timeout=60,
            config=default_regression,
        ).fit(X_train, y_train)

        before = model.feature_names_.copy()
        restored = pickle.loads(pickle.dumps(model))

        assert restored.feature_names_ == before
