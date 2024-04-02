"""
Tests for deep classifier trees.
"""
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from bezzanlabs.treemachine.deep_trees import DeepTreeClassifier, BaseDeep


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
def trained_model(classification_data) -> DeepTreeClassifier:
    X_train, _, y_train, _ = classification_data

    model = DeepTreeClassifier()
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def trained_multi(multiclass_data) -> DeepTreeClassifier:
    X_train, _, y_train, _ = multiclass_data

    model = DeepTreeClassifier()
    model.fit(X_train, y_train, batch_size=1000, epochs=10, verbose=1)
    return model


def test_model_predict(classification_data, trained_model):
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model.predict(X_test)))


def test_model_predict_proba(classification_data, trained_model):
    _, X_test, _, _ = classification_data
    assert all(np.isreal(trained_model.predict_proba(X_test).sum(axis=1)))


def test_model_predict_multi(multiclass_data, trained_multi):
    _, X_test, _, _ = multiclass_data
    assert all(np.isreal(trained_multi.predict(X_test)))


def test_model_score(classification_data, trained_model):
    _, X_test, _, y_test = classification_data
    assert trained_model.score(X_test, y_test)


@pytest.mark.skipif(
    BaseDeep._tf_version >= (2, 16, 0),
    reason="TF and shap are not compatible in 2.16",
)
def test_model_explain(classification_data, trained_model):
    _, X_test, _, _ = classification_data
    explain = trained_model.explain(X_test)

    assert explain[0][0].shape == X_test.shape
