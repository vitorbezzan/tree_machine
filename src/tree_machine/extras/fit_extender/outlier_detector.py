"""
Extends the .fit() method with an automatic outlier detector.
"""
import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from tree_machine.types import GroundTruth, Inputs

from .fit_extender import ExtenderResults


class OutlierDetector:
    """
    Extends estimator with an IsolationForest outlier detector.

    (Needs to copy whatever X it receives to work, so it may incur in some
    performance penalty).
    """

    OutlierDetector__column_name: str
    OutlierDetector__detector: IsolationForest

    def fit_(self, X: Inputs, y: GroundTruth, **extend_params) -> ExtenderResults:
        column_name = extend_params.get("column_name", "outlier_detector")

        detector = IsolationForest(**extend_params)
        detector.fit(X, y)

        return ExtenderResults(
            X=_create_column(column_name, detector.decision_function(X), X),
            y=y,
            attrs={"detector": detector, "column_name": column_name},
        )

    def predict_(self, X: Inputs) -> Inputs:
        return _create_column(
            self.OutlierDetector__column_name,
            self.OutlierDetector__detector.decision_function(X),
            X,
        )

    def score_(self, X: Inputs, y: GroundTruth) -> tuple[Inputs, GroundTruth]:
        return X, y


def _create_column(name: str, column: GroundTruth, X: Inputs) -> Inputs:
    X_ = copy.copy(X)
    if isinstance(X_, pd.DataFrame):
        X_[name] = column
    elif isinstance(X, np.ndarray):
        X_ = np.append(X_, column, axis=1)

    return X_
