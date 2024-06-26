"""
Classifier auto deep tree.
"""
import numpy as np
from imblearn.pipeline import Pipeline
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils.validation import check_is_fitted

from ..optimize import OptimizerCVMixIn
from ..transforms import Identity
from ..types import Actuals, Inputs, Predictions
from .config import defaults
from .deep_classifier import DeepTreeClassifier


class DeepClassifierCV(BaseEstimator, ClassifierMixin, OptimizerCVMixIn):
    """
    Implements DeepClassifierCV.
    """

    model_: DeepTreeClassifier

    def __init__(
        self,
        cv: BaseCrossValidator = KFold(n_splits=5),
        n_trials: int = 100,
        timeout: int = 180,
    ):
        """
        Constructor for DeepClassifierCV.
        """
        self.cv = cv
        self.setup(n_trials, timeout, True)

    def fit(self, X: Inputs, y: Actuals, **fit_params) -> "DeepClassifierCV":
        """
        Fits estimator using bayesian optimization.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass to the base
                classifier or the parameter distribution.
        """
        pipeline = Pipeline(
            [
                ("sampler", fit_params.pop("sampler", Identity())),
                ("estimator", DeepTreeClassifier()),
            ],
        )

        distributions = {
            f"estimator__{k}": v
            for k, v in fit_params.pop("distributions", defaults).items()
        }

        self._fit(
            pipeline,
            X,
            y,
            distributions,
            None,
            self.cv,
            **{f"estimator__{k}": v for k, v in fit_params.items()},
        )

        self.model_ = self.optimizer_.best_estimator_.steps[-1][1]

        return self

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model predictions.
        """
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Returns model probabilities.
        """
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def score(
        self,
        X: Inputs,
        y: Actuals,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Returns model score.
        """
        check_is_fitted(self, "model_")
        return self.model_.score(X, y, sample_weight=sample_weight)
