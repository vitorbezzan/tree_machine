"""Tests for pickle conformance with custom metric functions.

This module verifies that ClassifierCV and RegressionCV models with custom metric
functions can be pickled and unpickled while maintaining functionality.
"""

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
    """Return a regression train/test split as pandas DataFrames."""
    X, y = make_regression(
        n_samples=500, n_features=15, n_informative=10, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(15)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def classification_data():
    """Return a classification train/test split as pandas DataFrames."""
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=15, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(20)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


class TestClassifierCVPickleWithCustomMetrics:
    @pytest.fixture
    def trained_classifier_custom_accuracy(self, classification_data):
        X_train, _, y_train, _ = classification_data

        model = ClassifierCV(
            metric=custom_accuracy,
            cv=StratifiedKFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_classifier,
        )
        model.fit(X_train, y_train)
        return model

    @pytest.fixture
    def trained_classifier_custom_f1(self, classification_data):
        X_train, _, y_train, _ = classification_data

        model = ClassifierCV(
            metric=custom_f1,
            cv=StratifiedKFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_classifier,
        )
        model.fit(X_train, y_train)
        return model

    def test_pickle_bytes_with_custom_accuracy(
        self, trained_classifier_custom_accuracy
    ):
        pickled = pickle.dumps(trained_classifier_custom_accuracy)

        restored_model = pickle.loads(pickled)

        assert hasattr(restored_model, "model_")
        assert hasattr(restored_model, "metric")
        assert callable(restored_model.metric)

    def test_pickle_file_with_custom_f1(self, trained_classifier_custom_f1):
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_file = Path(tmpdir) / "classifier.pkl"

            with open(pickle_file, "wb") as f:
                pickle.dump(trained_classifier_custom_f1, f)

            with open(pickle_file, "rb") as f:
                restored_model = pickle.load(f)

            assert hasattr(restored_model, "model_")
            assert hasattr(restored_model, "metric")
            assert callable(restored_model.metric)

    def test_pickle_predictions_consistent_with_custom_accuracy(
        self, trained_classifier_custom_accuracy, classification_data
    ):
        _, X_test, _, _ = classification_data

        predictions_before = trained_classifier_custom_accuracy.predict(X_test)

        pickled = pickle.dumps(trained_classifier_custom_accuracy)
        restored_model = pickle.loads(pickled)

        predictions_after = restored_model.predict(X_test)

        np.testing.assert_array_equal(predictions_before, predictions_after)

    def test_pickle_proba_consistent_with_custom_f1(
        self, trained_classifier_custom_f1, classification_data
    ):
        _, X_test, _, _ = classification_data

        proba_before = trained_classifier_custom_f1.predict_proba(X_test)

        pickled = pickle.dumps(trained_classifier_custom_f1)
        restored_model = pickle.loads(pickled)

        proba_after = restored_model.predict_proba(X_test)

        np.testing.assert_array_almost_equal(proba_before, proba_after)

    def test_pickle_score_consistent_with_custom_accuracy(
        self, trained_classifier_custom_accuracy, classification_data
    ):
        _, X_test, _, y_test = classification_data

        score_before = trained_classifier_custom_accuracy.score(X_test, y_test)

        pickled = pickle.dumps(trained_classifier_custom_accuracy)
        restored_model = pickle.loads(pickled)

        score_after = restored_model.score(X_test, y_test)

        assert score_before == score_after

    def test_pickle_metric_is_callable_after_unpickle(
        self, trained_classifier_custom_accuracy
    ):
        pickled = pickle.dumps(trained_classifier_custom_accuracy)
        restored_model = pickle.loads(pickled)

        assert callable(restored_model.metric)

        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        result = restored_model.metric(y_true, y_pred)
        assert isinstance(result, (int, float, np.number))

    def test_pickle_feature_importances_preserved(
        self, trained_classifier_custom_accuracy
    ):
        feature_importances_before = (
            trained_classifier_custom_accuracy.feature_importances_.copy()
        )

        pickled = pickle.dumps(trained_classifier_custom_accuracy)
        restored_model = pickle.loads(pickled)

        np.testing.assert_array_equal(
            feature_importances_before,
            restored_model.feature_importances_,
        )

    def test_pickle_multiple_cycles(
        self, trained_classifier_custom_f1, classification_data
    ):
        _, X_test, _, _ = classification_data

        predictions_original = trained_classifier_custom_f1.predict(X_test)
        current_model = trained_classifier_custom_f1

        for _ in range(3):
            pickled = pickle.dumps(current_model)
            current_model = pickle.loads(pickled)

        predictions_final = current_model.predict(X_test)
        np.testing.assert_array_equal(predictions_original, predictions_final)


class TestRegressionCVPickleWithCustomMetrics:
    @pytest.fixture
    def trained_regressor_custom_r2(self, regression_data):
        X_train, _, y_train, _ = regression_data

        model = RegressionCV(
            metric=custom_r2,
            cv=KFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_regression,
        )
        model.fit(X_train, y_train)
        return model

    @pytest.fixture
    def trained_regressor_custom_mse(self, regression_data):
        X_train, _, y_train, _ = regression_data

        model = RegressionCV(
            metric=custom_mse,
            cv=KFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_regression,
        )
        model.fit(X_train, y_train)
        return model

    def test_pickle_bytes_with_custom_r2(self, trained_regressor_custom_r2):
        pickled = pickle.dumps(trained_regressor_custom_r2)

        restored_model = pickle.loads(pickled)

        assert hasattr(restored_model, "model_")
        assert hasattr(restored_model, "metric")
        assert callable(restored_model.metric)

    def test_pickle_file_with_custom_mse(self, trained_regressor_custom_mse):
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_file = Path(tmpdir) / "regressor.pkl"

            with open(pickle_file, "wb") as f:
                pickle.dump(trained_regressor_custom_mse, f)

            with open(pickle_file, "rb") as f:
                restored_model = pickle.load(f)

            assert hasattr(restored_model, "model_")
            assert hasattr(restored_model, "metric")
            assert callable(restored_model.metric)

    def test_pickle_predictions_consistent_with_custom_r2(
        self, trained_regressor_custom_r2, regression_data
    ):
        _, X_test, _, _ = regression_data

        predictions_before = trained_regressor_custom_r2.predict(X_test)

        pickled = pickle.dumps(trained_regressor_custom_r2)
        restored_model = pickle.loads(pickled)

        predictions_after = restored_model.predict(X_test)

        np.testing.assert_array_almost_equal(predictions_before, predictions_after)

    def test_pickle_score_consistent_with_custom_r2(
        self, trained_regressor_custom_r2, regression_data
    ):
        _, X_test, _, y_test = regression_data

        score_before = trained_regressor_custom_r2.score(X_test, y_test)

        pickled = pickle.dumps(trained_regressor_custom_r2)
        restored_model = pickle.loads(pickled)

        score_after = restored_model.score(X_test, y_test)

        assert abs(score_before - score_after) < 1e-10

    def test_pickle_score_consistent_with_custom_mse(
        self, trained_regressor_custom_mse, regression_data
    ):
        _, X_test, _, y_test = regression_data

        score_before = trained_regressor_custom_mse.score(X_test, y_test)

        pickled = pickle.dumps(trained_regressor_custom_mse)
        restored_model = pickle.loads(pickled)

        score_after = restored_model.score(X_test, y_test)

        assert abs(score_before - score_after) < 1e-10

    def test_pickle_metric_is_callable_after_unpickle(
        self, trained_regressor_custom_r2
    ):
        pickled = pickle.dumps(trained_regressor_custom_r2)
        restored_model = pickle.loads(pickled)

        assert callable(restored_model.metric)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        result = restored_model.metric(y_true, y_pred)
        assert isinstance(result, (int, float, np.number))

    def test_pickle_feature_importances_preserved(self, trained_regressor_custom_r2):
        feature_importances_before = (
            trained_regressor_custom_r2.feature_importances_.copy()
        )

        pickled = pickle.dumps(trained_regressor_custom_r2)
        restored_model = pickle.loads(pickled)

        np.testing.assert_array_equal(
            feature_importances_before,
            restored_model.feature_importances_,
        )

    def test_pickle_multiple_cycles(
        self, trained_regressor_custom_mse, regression_data
    ):
        _, X_test, _, _ = regression_data

        predictions_original = trained_regressor_custom_mse.predict(X_test)
        current_model = trained_regressor_custom_mse

        for _ in range(3):
            pickled = pickle.dumps(current_model)
            current_model = pickle.loads(pickled)

        predictions_final = current_model.predict(X_test)
        np.testing.assert_array_almost_equal(predictions_original, predictions_final)


class TestPickleWithFittedAttributes:
    def test_classifier_cv_attribute_preserved(self, classification_data):
        X_train, _, y_train, _ = classification_data

        model = ClassifierCV(
            metric=custom_accuracy,
            cv=StratifiedKFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_classifier,
        )
        model.fit(X_train, y_train)

        pickled = pickle.dumps(model)
        restored_model = pickle.loads(pickled)

        assert restored_model.cv is not None
        assert isinstance(restored_model.cv, StratifiedKFold)
        assert restored_model.cv.n_splits == 3

    def test_regressor_cv_attribute_preserved(self, regression_data):
        X_train, _, y_train, _ = regression_data

        model = RegressionCV(
            metric=custom_r2,
            cv=KFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_regression,
        )
        model.fit(X_train, y_train)

        pickled = pickle.dumps(model)
        restored_model = pickle.loads(pickled)

        assert restored_model.cv is not None
        assert isinstance(restored_model.cv, KFold)
        assert restored_model.cv.n_splits == 3

    def test_classifier_feature_names_preserved(self, classification_data):
        X_train, _, y_train, _ = classification_data

        model = ClassifierCV(
            metric=custom_f1,
            cv=StratifiedKFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_classifier,
        )
        model.fit(X_train, y_train)

        feature_names_before = model.feature_names_.copy()

        pickled = pickle.dumps(model)
        restored_model = pickle.loads(pickled)

        assert restored_model.feature_names_ == feature_names_before

    def test_regressor_feature_names_preserved(self, regression_data):
        X_train, _, y_train, _ = regression_data

        model = RegressionCV(
            metric=custom_mse,
            cv=KFold(n_splits=3),
            n_trials=5,
            timeout=60,
            config=default_regression,
        )
        model.fit(X_train, y_train)

        feature_names_before = model.feature_names_.copy()

        pickled = pickle.dumps(model)
        restored_model = pickle.loads(pickled)

        assert restored_model.feature_names_ == feature_names_before
