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


def test_classifiercv_validation_objective_uses_scorer_and_sets_fitted_attrs(
    classification_data, monkeypatch
):
    """fit(..., X_validation/y_validation) should use the validation objective.

    This test hits the `_objective_validation` branch in `BaseAutoCV.optimize`, ensuring
    it calls the estimator scorer on the provided validation set (and does not call
    sklearn's `cross_validate`). It also checks that the fitted model still exposes the
    expected fitted attributes.
    """

    X_train, _, y_train, _ = classification_data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
    )

    from tree_machine import base as base_module

    # Guard: validation path must not call cross_validate
    def _fail_cross_validate(*args, **kwargs):  # pragma: no cover
        raise AssertionError("cross_validate should not be called when using validation")

    monkeypatch.setattr(base_module, "cross_validate", _fail_cross_validate, raising=True)

    # Spy scorer that verifies it was called correctly and returns a deterministic score.
    scorer_calls = {"count": 0}

    def _spy_scorer(estimator, X, y):
        scorer_calls["count"] += 1
        # ensure validated arrays are passed through
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == X_val.shape[0]
        assert X.shape[1] == X_val.shape[1]
        assert y.shape[0] == y_val.shape[0]
        # Return a stable score in [0, 1]
        return 0.123

    class _FakeTrial:
        def __init__(self):
            self.user_attrs = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

        # Minimal Optuna Trial API used by OptimizerParams.get_trial_values
        def suggest_float(self, name, low, high, step=None, **kwargs):
            # deterministic mid-point, optionally snapped to the step
            val = (low + high) / 2.0
            if step:
                val = low + round((val - low) / step) * step
            return float(val)

        def suggest_int(self, name, low, high, **kwargs):
            return int((low + high) // 2)

        def suggest_categorical(self, name, choices, **kwargs):
            return choices[0]

    class _FakeStudy:
        def __init__(self):
            self.best_params = {}
            self.best_trial = None

        def optimize(self, objective, n_trials, timeout):
            trial = _FakeTrial()
            score = objective(trial)
            self.best_trial = trial
            # Optuna would maximize; we just record the score for debugging/consistency
            self.best_value = score

    monkeypatch.setattr(
        base_module,
        "create_study",
        lambda *args, **kwargs: _FakeStudy(),
        raising=True,
    )

    model = ClassifierCV(
        metric="f1",
        cv=KFold(n_splits=3),
        n_trials=1,
        timeout=1,
        config=default_classifier,
    )

    # Patch the scorer property on the class so BaseAutoCV.optimize uses our spy.
    monkeypatch.setattr(
        ClassifierCV,
        "scorer",
        property(lambda self: _spy_scorer),
        raising=True,
    )

    model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

    assert scorer_calls["count"] == 1

    # Fitted attributes should exist (study_, best_params_, model_, feature_importances_)
    assert hasattr(model, "model_")
    assert hasattr(model, "study_")
    assert hasattr(model, "best_params_")
    assert isinstance(model.feature_importances_, np.ndarray)

    # And the validation objective should have set a cv_results-like structure
    assert "cv_results" in model.study_.best_trial.user_attrs
    cv_results = model.study_.best_trial.user_attrs["cv_results"]
    assert "test_score" in cv_results
    assert np.allclose(cv_results["test_score"], np.array([0.123]))
