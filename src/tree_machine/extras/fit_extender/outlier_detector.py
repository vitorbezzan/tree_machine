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

    outlier_detector: IsolationForest

    def pre_fit(self, X: Inputs, y: GroundTruth, **extend_params) -> ExtenderResults:
        """Captures .fit() data and returns treated data and attrs."""
        detector = IsolationForest(**extend_params)
        detector.fit(X, y)

        X_ = X.copy()

        if isinstance(X, pd.DataFrame):
            X_["outlier_detector"] = detector.decision_function(X)
        elif isinstance(X, np.ndarray):
            X_ = np.append(X, detector.decision_function(X), axis=1)
        else:
            raise NotImplementedError()

        return ExtenderResults(
            X=X_,
            y=y,
            attrs={"outlier_detector": detector},
        )

    def pre_predict(self, X: Inputs) -> Inputs:
        """Captures .predict() data and returns treated data."""
        X_ = X.copy()
        if isinstance(X, pd.DataFrame):
            X_["outlier_detector"] = self.outlier_detector.decision_function(X)
        elif isinstance(X, np.ndarray):
            X_ = np.append(X, self.outlier_detector.decision_function(X), axis=1)
        else:
            raise NotImplementedError()

        return X_
