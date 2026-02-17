"""Tests for classifier cross-validated estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, train_test_split

from tree_machine import ClassifierCV, ClassifierCVConfig, default_classifier


@pytest.fixture(scope="session")
def classification_data():
    """Return a binary classification train/test split as pandas DataFrames."""
    X, y = make_classification(n_samples=1000, n_features=30, n_informative=20)
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(30)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def multiclass_data():
    """Return a multiclass train/test split as pandas DataFrames."""
    X, y = make_classification(
        n_samples=2000, n_features=30, n_informative=20, n_classes=4
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(30)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def trained_model(classification_data) -> ClassifierCV:
    """Fit a default binary ClassifierCV model."""
    X_train, _, y_train, _ = classification_data

    model = ClassifierCV(
        metric="f1",
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=default_classifier,
    ).fit(
        X_train,
        y_train,
    )
    return model


@pytest.fixture(scope="session")
def trained_multi(multiclass_data) -> ClassifierCV:
    """Fit a default multiclass ClassifierCV model."""
    X_train, _, y_train, _ = multiclass_data

    model = ClassifierCV(
        metric="f1_micro",
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=default_classifier,
    ).fit(
        X_train,
        y_train,
    )
    return model


def test_model_predict(classification_data, trained_model):
    """predict should return finite numeric labels."""
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model.predict(X_test)))


def test_model_predict_multi(multiclass_data, trained_multi):
    """predict should work for multiclass models."""
    _, X_test, _, _ = multiclass_data
    assert all(np.isreal(trained_multi.predict(X_test.values)))


def test_model_predict_proba(classification_data, trained_model):
    """predict_proba should return probabilities that sum to 1."""
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model.predict_proba(X_test).sum(axis=1)))


def test_model_score(classification_data, trained_model):
    """score should return a truthy value for a fitted model."""
    _, X_test, _, y_test = classification_data
    assert trained_model.score(X_test, y_test)


def test_model_explain(classification_data, trained_model):
    """explain should return SHAP values with expected shape."""
    _, X_test, _, _ = classification_data

    explain = trained_model.explain(X_test)
    assert explain["shap_values"].shape == (250, 30, 1)


def test_model_explain_multi(multiclass_data, trained_multi):
    """explain should support multiclass models."""
    _, X_test, _, _ = multiclass_data

    explain = trained_multi.explain(X_test)
    assert explain["shap_values"].shape == (500, 30, 4)


def test_model_performance(classification_data, trained_model):
    """Trained model should outperform a dummy baseline."""
    X_train, X_test, y_train, y_test = classification_data

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    baseline_score = dummy.score(y_test, y_test)
    model_score = trained_model.score(X_test, y_test)

    assert baseline_score < model_score


@pytest.fixture(scope="session")
def trained_model_catboost(classification_data) -> ClassifierCV:
    """Fit a binary ClassifierCV model using the CatBoost backend."""
    X_train, _, y_train, _ = classification_data

    config = ClassifierCVConfig(
        monotone_constraints={},
        interactions=[],
        n_jobs=1,
        parameters=default_classifier.parameters,
        return_train_score=True,
    )

    model = ClassifierCV(
        metric="f1",
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=config,
        backend="catboost",
    ).fit(
        X_train,
        y_train,
    )
    return model


@pytest.fixture(scope="session")
def trained_multi_catboost(multiclass_data) -> ClassifierCV:
    """Fit a multiclass ClassifierCV model using the CatBoost backend."""
    X_train, _, y_train, _ = multiclass_data

    config = ClassifierCVConfig(
        monotone_constraints={},
        interactions=[],
        n_jobs=1,
        parameters=default_classifier.parameters,
        return_train_score=True,
    )

    model = ClassifierCV(
        metric="f1_micro",
        cv=KFold(n_splits=5),
        n_trials=50,
        timeout=120,
        config=config,
        backend="catboost",
    ).fit(
        X_train,
        y_train,
    )
    return model


def test_model_predict_catboost(classification_data, trained_model_catboost):
    """predict should work for the CatBoost backend."""
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model_catboost.predict(X_test)))


def test_model_predict_multi_catboost(multiclass_data, trained_multi_catboost):
    """predict should work for the CatBoost backend in multiclass mode."""
    _, X_test, _, _ = multiclass_data
    assert all(np.isreal(trained_multi_catboost.predict(X_test.values)))


def test_model_predict_proba_catboost(classification_data, trained_model_catboost):
    """predict_proba should work for the CatBoost backend."""
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model_catboost.predict_proba(X_test).sum(axis=1)))


def test_model_score_catboost(classification_data, trained_model_catboost):
    """score should work for the CatBoost backend."""
    _, X_test, _, y_test = classification_data
    assert trained_model_catboost.score(X_test, y_test)


def test_model_explain_catboost(classification_data, trained_model_catboost):
    """explain should work for the CatBoost backend."""
    _, X_test, _, _ = classification_data

    explain = trained_model_catboost.explain(X_test)
    assert explain["shap_values"].shape == (250, 30, 1)


def test_model_explain_multi_catboost(multiclass_data, trained_multi_catboost):
    """explain should work for multiclass CatBoost models."""
    _, X_test, _, _ = multiclass_data

    explain = trained_multi_catboost.explain(X_test)
    assert explain["shap_values"].shape == (500, 4, 30)


def test_model_performance_catboost(classification_data, trained_model_catboost):
    """CatBoost-backed model should outperform a dummy baseline."""
    X_train, X_test, y_train, y_test = classification_data

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    baseline_score = dummy.score(y_test, y_test)
    model_score = trained_model_catboost.score(X_test, y_test)

    assert baseline_score < model_score


def test_classifiercv_forwards_validation_fit_params_to_optimize(
    classification_data, monkeypatch
):
    """fit(**fit_params) should forward X_validation/y_validation into BaseAutoCV.optimize."""
    X_train, _, y_train, _ = classification_data

    # Use a small explicit validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
    )

    from tree_machine.base import BaseAutoCV

    def _spy_optimize(self, *args, **kwargs):
        assert kwargs.get("X_validation") is X_val
        assert kwargs.get("y_validation") is y_val

        # Return something that looks like a fitted model for the rest of fit()
        class _DummyModel:
            feature_importances_ = np.array([], dtype=float)

        return _DummyModel()

    monkeypatch.setattr(BaseAutoCV, "optimize", _spy_optimize, raising=True)

    model = ClassifierCV(
        metric="f1",
        cv=KFold(n_splits=3),
        n_trials=1,
        timeout=1,
        config=default_classifier,
    )

    model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)
    assert hasattr(model, "model_")


def test_classifiercv_uses_validation_set_for_optimization(classification_data):
    """When validation data is provided, optimization should use validation scoring, not CV."""
    X_train, _, y_train, _ = classification_data

    # Create explicit train/validation split with stratification
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Fit model with validation data
    model = ClassifierCV(
        metric="f1",
        cv=KFold(n_splits=3),
        n_trials=3,  # Use multiple trials to ensure optimization works
        timeout=10,
        config=default_classifier,
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

    # Verify predicted probabilities
    probabilities = model.predict_proba(X_val)
    assert probabilities.shape == (len(y_val), 2)  # Binary classification
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    # Verify the score is reasonable (model produces a valid F1 score)
    score = model.score(X_val, y_val)
    assert score > 0.0  # F1 should be positive and finite for a valid model
    assert score <= 1.0  # F1 is bounded by 1.0
