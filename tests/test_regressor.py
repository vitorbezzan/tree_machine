"""
Tests for regressor trees.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, train_test_split

from bezzanlabs.treemachine import Regressor
from bezzanlabs.treemachine.trees.config import regression_metrics


@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=20)
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(20)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def trained_model(regression_data):
    X_train, _, y_train, _ = regression_data

    model = Regressor(metric="mse", split=KFold(n_splits=5)).fit(
        X_train,
        y_train,
    )
    return model


def test_model_predict(regression_data, trained_model):
    _, X_test, _, _ = regression_data
    assert all(np.isreal(trained_model.predict(X_test)))


def test_model_score(regression_data, trained_model):
    _, X_test, _, y_test = regression_data
    assert trained_model.score(X_test, y_test)


def test_model_explain(regression_data, trained_model):
    _, X_test, _, _ = regression_data

    explain = trained_model.explain(X_test)
    assert explain[0].shape == (250, 20)


def test_model_performance(regression_data, trained_model):
    X_train, X_test, y_train, y_test = regression_data

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)

    baseline_score = -regression_metrics["mse"](y_test, dummy.predict(X_test))
    model_score = trained_model.score(X_test, y_test)

    assert baseline_score < model_score
