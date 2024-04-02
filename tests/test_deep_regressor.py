"""
Tests for regressor trees.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from bezzanlabs.treemachine.deep_trees import DeepTreeRegressor, BaseDeep


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

    model = DeepTreeRegressor().fit(X_train, y_train)
    return model


def test_model_predict(regression_data, trained_model):
    _, X_test, _, _ = regression_data
    assert all(np.isreal(trained_model.predict(X_test)))


def test_model_score(regression_data, trained_model):
    _, X_test, _, y_test = regression_data
    assert trained_model.score(X_test, y_test)


@pytest.mark.skipif(
    BaseDeep._tf_version >= (2, 16, 0),
    reason="TF and shap are not compatible in 2.16",
)
def test_model_explain(regression_data, trained_model):
    _, X_test, _, _ = regression_data
    explain = trained_model.explain(X_test)

    assert explain[0][0].shape == X_test.shape
