"""
Extends the .fit() method with a way to include the training data directly in the
estimator.
"""
from tree_machine.types import GroundTruth, Inputs

from .fit_extender import ExtenderResults


class TrainData:
    """Includes training data for an estimator in the estimator itself."""

    def fit_(self, X: Inputs, y: GroundTruth, **extend_params) -> ExtenderResults:
        return ExtenderResults(
            X=X,
            y=y,
            attrs={"X": X, "y": y},
        )

    def predict_(self, X: Inputs) -> Inputs:
        return X

    def score_(self, X: Inputs, y: GroundTruth) -> tuple[Inputs, GroundTruth]:
        return X, y
