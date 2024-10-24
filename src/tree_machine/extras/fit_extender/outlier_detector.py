"""
Extends the .fit() method with an automatic outlier detector.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from tree_machine.types import GroundTruth, Inputs

from .fit_extender import ExtenderResults


class OutlierDetector:
    """Outlier detector extension for models."""

    OutlierDetector__detector: IsolationForest

    def pre_fit(self, X: Inputs, y: GroundTruth, **extend_params) -> ExtenderResults:
        """Captures .fit() data and returns treated data and attrs."""
        name = extend_params.get("column_name", "outlier_detector")

        detector = IsolationForest(**extend_params)
        detector.fit(X, y)

        X_ = X.copy()
        if isinstance(X, pd.DataFrame):
            X_[name] = detector.decision_function(X)
        elif isinstance(X, np.ndarray):
            X_ = np.append(X_, detector.decision_function(X), axis=1)
        else:
            raise NotImplementedError()

        return ExtenderResults(
            X=X_,
            y=y,
            attrs={"detector": detector, "column_name": name},
        )

    def pre_predict(self, X: Inputs) -> Inputs:
        """Captures .predict() data and returns treated data."""
        column_name = getattr(self, "OutlierDetector__column_name")

        X_ = X.copy()
        if isinstance(X, pd.DataFrame):
            X_[column_name] = self.OutlierDetector__detector.decision_function(X)
        elif isinstance(X, np.ndarray):
            X_ = np.append(
                X_, self.OutlierDetector__detector.decision_function(X), axis=1
            )
        else:
            raise NotImplementedError()

        return X_
