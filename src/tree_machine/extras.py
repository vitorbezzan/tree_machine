"""
Extras for training models.
"""
import typing as tp
from abc import ABC, abstractmethod

from sklearn.ensemble import IsolationForest

from tree_machine.types import GroundTruth, Inputs


@tp.runtime_checkable
class HaveFit(tp.Protocol):
    def fit(self, X: Inputs, y: GroundTruth, **fit_params) -> "HaveFit":
        ...


class FitExtender(ABC):
    parameters: set[str]

    @staticmethod
    @abstractmethod
    def extend(X: Inputs, y: GroundTruth, **extend_params) -> dict[str, object]:
        raise NotImplementedError()


def fit_extend(estimator: tp.Type, extender: tp.Type[FitExtender]) -> tp.Type:
    """Extends the .fit() functionality for an estimator."""
    if issubclass(estimator, HaveFit):
        if issubclass(extender, FitExtender) and not issubclass(extender, HaveFit):

            class Extended(estimator):
                def fit(self, X: Inputs, y: GroundTruth, **fit_params):
                    """Re-implements .fit() method for extended class."""
                    captured = {
                        k: v for k, v in fit_params.items() if k in extender.parameters
                    }

                    not_captured = {
                        k: v
                        for k, v in fit_params.items()
                        if k not in extender.parameters
                    }

                    for attrib, value in extender.extend(X, y, **captured).items():
                        setattr(self, attrib, value)

                    return super().fit(X, y, **not_captured)

            return Extended

    raise TypeError("estimator/extender type mismatch.")


class OutlierDetector(FitExtender):
    """
    Implements a simple outlier detector that is fitted at the same time as a model.
    """

    parameters = {}

    @staticmethod
    @abstractmethod
    def extend(X: Inputs, y: GroundTruth, **extend_params) -> dict[str, object]:
        return {
            "outlier_detector": IsolationForest().fit(X, y),
        }
