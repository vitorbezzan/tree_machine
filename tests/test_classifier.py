"""
Tests for classifier trees.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, train_test_split

from tree_machine import ClassifierCV, ClassifierCVConfig, default_classifier


@pytest.fixture(scope="session")
def classification_data():
    X, y = make_classification(n_samples=1000, n_features=30, n_informative=20)
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(30)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def multiclass_data():
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
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model.predict(X_test)))


def test_model_predict_multi(multiclass_data, trained_multi):
    _, X_test, _, _ = multiclass_data
    assert all(np.isreal(trained_multi.predict(X_test.values)))


def test_model_predict_proba(classification_data, trained_model):
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model.predict_proba(X_test).sum(axis=1)))


def test_model_score(classification_data, trained_model):
    _, X_test, _, y_test = classification_data
    assert trained_model.score(X_test, y_test)


def test_model_explain(classification_data, trained_model):
    _, X_test, _, _ = classification_data

    explain = trained_model.explain(X_test)
    assert explain["shap_values"].shape == (250, 30, 1)


def test_model_explain_multi(multiclass_data, trained_multi):
    _, X_test, _, _ = multiclass_data

    explain = trained_multi.explain(X_test)
    assert explain["shap_values"].shape == (500, 30, 4)


def test_model_performance(classification_data, trained_model):
    X_train, X_test, y_train, y_test = classification_data

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    baseline_score = dummy.score(y_test, y_test)
    model_score = trained_model.score(X_test, y_test)

    assert baseline_score < model_score


@pytest.fixture(scope="session")
def trained_model_catboost(classification_data) -> ClassifierCV:
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
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model_catboost.predict(X_test)))


def test_model_predict_multi_catboost(multiclass_data, trained_multi_catboost):
    _, X_test, _, _ = multiclass_data
    assert all(np.isreal(trained_multi_catboost.predict(X_test.values)))


def test_model_predict_proba_catboost(classification_data, trained_model_catboost):
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model_catboost.predict_proba(X_test).sum(axis=1)))


def test_model_score_catboost(classification_data, trained_model_catboost):
    _, X_test, _, y_test = classification_data
    assert trained_model_catboost.score(X_test, y_test)


def test_model_explain_catboost(classification_data, trained_model_catboost):
    _, X_test, _, _ = classification_data

    explain = trained_model_catboost.explain(X_test)
    assert explain["shap_values"].shape == (250, 30, 1)


def test_model_explain_multi_catboost(multiclass_data, trained_multi_catboost):
    _, X_test, _, _ = multiclass_data

    explain = trained_multi_catboost.explain(X_test)
    assert explain["shap_values"].shape == (500, 4, 30)


def test_model_performance_catboost(classification_data, trained_model_catboost):
    X_train, X_test, y_train, y_test = classification_data

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    baseline_score = dummy.score(y_test, y_test)
    model_score = trained_model_catboost.score(X_test, y_test)

    assert baseline_score < model_score
