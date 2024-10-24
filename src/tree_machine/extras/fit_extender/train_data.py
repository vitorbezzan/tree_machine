"""
Extends the .fit() method with a way to include the training data directly in the
estimator.
"""
from tree_machine.types import GroundTruth, Inputs

from .fit_extender import ExtenderResults


class TrainData:
    """Includes training data for an estimator in the estimator itself."""

    def pre_fit(self, X: Inputs, y: GroundTruth, **extend_params) -> ExtenderResults:
        """Captures .fit() data and returns treated data and attrs."""
        return ExtenderResults(
            X=X,
            y=y,
            attrs={"X": X, "y": y},
        )

    def pre_predict(self, X: Inputs) -> Inputs:
        """Captures .predict() data and returns treated data."""
        return X
