"""Tests for ClassifierCV across binary and multiclass cases."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, train_test_split

from tree_machine import ClassifierCV, ClassifierCVConfig, default_classifier


@pytest.fixture(scope="session")
def binary_data():
    """Return a binary classification train/test split."""
    X, y = make_classification(
        n_samples=1000, n_features=30, n_informative=20, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(30)])
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="session")
def multiclass_data():
    """Return a multiclass classification train/test split."""
    X, y = make_classification(
        n_samples=2000, n_features=30, n_informative=20, n_classes=4, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(30)])
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="session")
def validation_split(binary_data):
    """Provide a validation split derived from the binary dataset."""
    X_train, _, y_train, _ = binary_data
    return train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
    )


@pytest.fixture(scope="session")
def cv():
    """Return a reusable five-fold splitter."""
    return KFold(n_splits=5)


@pytest.fixture(scope="session")
def catboost_config():
    """Return ClassifierCV configuration for CatBoost backend."""
    return ClassifierCVConfig(
        monotone_constraints={},
        interactions=[],
        n_jobs=1,
        parameters=default_classifier.parameters,
        return_train_score=True,
    )


@pytest.fixture(scope="session")
def binary_model(binary_data, cv):
    """Train a binary ClassifierCV model."""
    X_train, _, y_train, _ = binary_data
    return ClassifierCV(
        metric="f1", cv=cv, n_trials=50, timeout=120, config=default_classifier
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def multiclass_model(multiclass_data, cv):
    """Train a multiclass ClassifierCV model."""
    X_train, _, y_train, _ = multiclass_data
    return ClassifierCV(
        metric="f1_micro", cv=cv, n_trials=50, timeout=120, config=default_classifier
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def binary_catboost(binary_data, cv, catboost_config):
    """Train a binary CatBoost-backed ClassifierCV model."""
    X_train, _, y_train, _ = binary_data
    return ClassifierCV(
        metric="f1",
        cv=cv,
        n_trials=50,
        timeout=120,
        config=catboost_config,
        backend="catboost",
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def multiclass_catboost(multiclass_data, cv, catboost_config):
    """Train a multiclass CatBoost-backed ClassifierCV model."""
    X_train, _, y_train, _ = multiclass_data
    return ClassifierCV(
        metric="f1_micro",
        cv=cv,
        n_trials=50,
        timeout=120,
        config=catboost_config,
        backend="catboost",
    ).fit(X_train, y_train)


@pytest.fixture(scope="session")
def dummy_baseline(binary_data):
    """Fit a dummy classifier baseline."""
    X_train, _, y_train, _ = binary_data
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    return dummy


@pytest.fixture(scope="session")
def binary_predictions(binary_data, binary_model):
    """Return binary predictions for the held-out set."""
    _, X_test, _, _ = binary_data
    return binary_model.predict(X_test)


@pytest.fixture(scope="session")
def binary_proba(binary_data, binary_model):
    """Return binary class probabilities for the held-out set."""
    _, X_test, _, _ = binary_data
    return binary_model.predict_proba(X_test)


@pytest.fixture(scope="session")
def binary_explanation(binary_data, binary_model):
    """Explain the binary model predictions."""
    _, X_test, _, _ = binary_data
    return binary_model.explain(X_test)


@pytest.fixture(scope="session")
def multiclass_explanation(multiclass_data, multiclass_model):
    """Explain the multiclass model predictions."""
    _, X_test, _, _ = multiclass_data
    return multiclass_model.explain(X_test)


@pytest.fixture(scope="session")
def catboost_predictions(binary_data, binary_catboost):
    """Return binary predictions from the CatBoost model."""
    _, X_test, _, _ = binary_data
    return binary_catboost.predict(X_test)


@pytest.fixture(scope="session")
def catboost_proba(binary_data, binary_catboost):
    """Return binary class probabilities from the CatBoost model."""
    _, X_test, _, _ = binary_data
    return binary_catboost.predict_proba(X_test)


@pytest.fixture(scope="session")
def catboost_explanation(binary_data, binary_catboost):
    """Explain the binary CatBoost model predictions."""
    _, X_test, _, _ = binary_data
    return binary_catboost.explain(X_test)


@pytest.fixture(scope="session")
def catboost_multiclass_explanation(multiclass_data, multiclass_catboost):
    """Explain the multiclass CatBoost model predictions."""
    _, X_test, _, _ = multiclass_data
    return multiclass_catboost.explain(X_test)


class TestBinaryClassifier:
    """Binary classifier behavior checks."""

    def test_predict_returns_real(self, binary_predictions):
        """Predictions should be real-valued."""
        assert all(np.isreal(binary_predictions))

    def test_proba_sums_to_one(self, binary_proba):
        """Class probabilities should sum to one per row."""
        assert np.allclose(binary_proba.sum(axis=1), 1.0)

    def test_score(self, binary_data, binary_model):
        """Model should achieve positive score."""
        _, X_test, _, y_test = binary_data
        assert binary_model.score(X_test, y_test) > 0

    def test_beats_baseline(self, binary_data, binary_model, dummy_baseline):
        """Model score should beat a dummy baseline."""
        _, X_test, _, y_test = binary_data
        assert binary_model.score(X_test, y_test) > dummy_baseline.score(X_test, y_test)

    def test_explain_shape(self, binary_explanation):
        """Explanation output should have expected shape."""
        assert binary_explanation["shap_values"].shape == (250, 30, 1)


class TestMulticlassClassifier:
    """Multiclass classifier behavior checks."""

    def test_predict_returns_real(self, multiclass_data, multiclass_model):
        """Predictions should be real-valued for multiclass model."""
        _, X_test, _, _ = multiclass_data
        assert all(np.isreal(multiclass_model.predict(X_test.values)))

    def test_explain_shape(self, multiclass_explanation):
        """Explanation output should have expected multiclass shape."""
        assert multiclass_explanation["shap_values"].shape == (500, 30, 4)


class TestCatBoostBinary:
    """Binary classifier using CatBoost backend."""

    def test_predict_returns_real(self, catboost_predictions):
        """CatBoost predictions should be real-valued."""
        assert all(np.isreal(catboost_predictions))

    def test_proba_sums_to_one(self, catboost_proba):
        """CatBoost probabilities should sum to one per row."""
        assert np.allclose(catboost_proba.sum(axis=1), 1.0)

    def test_score(self, binary_data, binary_catboost):
        """CatBoost model should achieve positive score."""
        _, X_test, _, y_test = binary_data
        assert binary_catboost.score(X_test, y_test) > 0

    def test_beats_baseline(self, binary_data, binary_catboost, dummy_baseline):
        """CatBoost model should beat the dummy baseline."""
        _, X_test, _, y_test = binary_data
        assert binary_catboost.score(X_test, y_test) > dummy_baseline.score(
            X_test, y_test
        )

    def test_explain_shape(self, catboost_explanation):
        """CatBoost explanation output should have expected shape."""
        assert catboost_explanation["shap_values"].shape == (250, 30, 1)


class TestCatBoostMulticlass:
    """Multiclass CatBoost classifier behavior checks."""

    def test_predict_returns_real(self, multiclass_data, multiclass_catboost):
        """CatBoost multiclass predictions should be real-valued."""
        _, X_test, _, _ = multiclass_data
        assert all(np.isreal(multiclass_catboost.predict(X_test.values)))

    def test_explain_shape(self, catboost_multiclass_explanation):
        """CatBoost multiclass explanation output should have expected shape."""
        assert catboost_multiclass_explanation["shap_values"].shape == (500, 4, 30)


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

        model = ClassifierCV(
            metric="f1",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_classifier,
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

        model = ClassifierCV(
            metric="f1",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_classifier,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert hasattr(model, "model_")


class TestValidationObjective:
    """Validation objective behavior for ClassifierCV."""

    @pytest.fixture
    def fake_study_setup(self, monkeypatch):
        """Provide a fake Optuna study and scorer spy."""
        from tree_machine import base as base_module

        scorer_calls = {"count": 0}

        def fail_cross_validate(*args, **kwargs):
            """Fail if cross_validate is mistakenly invoked."""
            raise AssertionError("cross_validate should not be called")

        def spy_scorer(estimator, X, y):
            """Increment a counter and return a dummy score."""
            scorer_calls["count"] += 1
            return 0.123

        class FakeTrial:
            """Minimal Optuna trial stub."""

            def __init__(self):
                self.user_attrs = {}

            def set_user_attr(self, key, value):
                """Store a user attribute."""
                self.user_attrs[key] = value

            def suggest_float(self, name, low, high, step=None, **kwargs):
                """Return a midpoint float suggestion respecting step."""
                val = (low + high) / 2.0
                if step:
                    val = low + round((val - low) / step) * step
                return float(val)

            def suggest_int(self, name, low, high, **kwargs):
                """Return a midpoint integer suggestion."""
                return int((low + high) // 2)

            def suggest_categorical(self, name, choices, **kwargs):
                """Return the first categorical choice."""
                return choices[0]

        class FakeStudy:
            """Minimal Optuna study stub."""

            def __init__(self):
                self.best_params = {}
                self.best_trial = None

            def optimize(self, objective, n_trials, timeout):
                """Optimize once and capture best trial/value."""
                trial = FakeTrial()
                score = objective(trial)
                self.best_trial = trial
                self.best_value = score

        monkeypatch.setattr(base_module, "cross_validate", fail_cross_validate)
        monkeypatch.setattr(base_module, "create_study", lambda *a, **kw: FakeStudy())
        monkeypatch.setattr(ClassifierCV, "scorer", property(lambda self: spy_scorer))

        return scorer_calls

    def test_uses_scorer(self, validation_split, fake_study_setup):
        """Scorer should be invoked once during validation fit."""
        X_tr, X_val, y_tr, y_val = validation_split

        model = ClassifierCV(
            metric="f1",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_classifier,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert fake_study_setup["count"] == 1

    def test_sets_fitted_attributes(self, validation_split, fake_study_setup):
        """Fitted model should expose fitted attributes after validation."""
        X_tr, X_val, y_tr, y_val = validation_split

        model = ClassifierCV(
            metric="f1",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_classifier,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert hasattr(model, "model_")
        assert hasattr(model, "study_")
        assert hasattr(model, "best_params_")

    def test_feature_importances(self, validation_split, fake_study_setup):
        """Feature importances should be a numpy array."""
        X_tr, X_val, y_tr, y_val = validation_split

        model = ClassifierCV(
            metric="f1",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_classifier,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        assert isinstance(model.feature_importances_, np.ndarray)

    def test_cv_results_structure(self, validation_split, fake_study_setup):
        """cv_results user attr should contain test_score key."""
        X_tr, X_val, y_tr, y_val = validation_split

        model = ClassifierCV(
            metric="f1",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_classifier,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        cv_results = model.study_.best_trial.user_attrs["cv_results"]
        assert "test_score" in cv_results

    def test_cv_results_score_value(self, validation_split, fake_study_setup):
        """cv_results test_score should match scorer output."""
        X_tr, X_val, y_tr, y_val = validation_split

        model = ClassifierCV(
            metric="f1",
            cv=KFold(n_splits=3),
            n_trials=1,
            timeout=1,
            config=default_classifier,
        )
        model.fit(X_tr, y_tr, X_validation=X_val, y_validation=y_val)

        cv_results = model.study_.best_trial.user_attrs["cv_results"]
        assert np.allclose(cv_results["test_score"], np.array([0.123]))
