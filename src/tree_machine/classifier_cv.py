# isort: skip_file
"""
Definition for ClassifierCV.
"""
import typing as tp

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import NonNegativeInt, validate_call
from pydantic.dataclasses import dataclass
from sklearn.base import ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from .base import BaseAutoCV
from .classification_metrics import AcceptableClassifier, classification_metrics
from .optimizer_params import OptimizerParams
from .types import GroundTruth, Inputs, Predictions


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class ClassifierCVConfig:
    """
    Available config to use when fitting a classification model.

    monotone_constraints: dictionary containing monotonicity direction allowed for each
        variable. 0 means no monotonicity, 1 means increasing and -1 means decreasing
        monotonicity.
    interactions: list of lists containing permitted relationships in data.
    parameters: dictionary with distribution bounds for each hyperparameter to search
        on during optimization.
    n_jobs: Number of jobs to use when fitting the model.
    sampler: `imblearn` sampler to use when fitting models.
    """

    monotone_constraints: dict[str, int]
    interactions: list[list[str]]
    n_jobs: int
    parameters: OptimizerParams

    def get_kwargs(self, feature_names: list[str]) -> dict:
        """
        Returns parsed and validated constraint configuration for a ClassifierCV model.

        Args:
            feature_names: list of feature names. If empty, will return empty
                constraints dictionaries and lists.
        """
        return {
            "monotone_constraints": {
                feature_names.index(key): value
                for key, value in self.monotone_constraints.items()
            },
            "interaction_constraints": [
                [feature_names.index(key) for key in lt] for lt in self.interactions
            ],
            "n_jobs": self.n_jobs,
        }


default_classifier = ClassifierCVConfig(
    monotone_constraints={},
    interactions=[],
    n_jobs=-1,
    parameters=OptimizerParams(),
)


class ClassifierCV(BaseAutoCV, ClassifierMixin):
    """
    Defines an auto classification tree, based on the bayesian optimization base class.
    """

    model_: XGBClassifier
    feature_importances_: NDArray[np.float64]

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: AcceptableClassifier,
        cv: BaseCrossValidator,
        n_trials: NonNegativeInt,
        timeout: NonNegativeInt,
        config: ClassifierCVConfig,
    ) -> None:
        """
        Constructor for ClassifierCV.

        Args:
            metric: Loss metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            config: Configuration to use when fitting the model.
        """
        super().__init__(metric, cv, n_trials, timeout)
        self._config = config

    def fit(self, X: Inputs, y: GroundTruth, **fit_params) -> "ClassifierCV":
        """
        Fits ClassifierCV.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
        """
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else []
        constraints = self._config.get_kwargs(self.feature_names_)

        self.model_ = self.optimize(
            estimator_type=XGBClassifier,
            X=self._validate_X(X),
            y=self._validate_y(y),
            parameters=self._config.parameters,
            **constraints,
        )
        self.feature_importances_ = self.model_.feature_importances_

        return self

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model predictions.
        """
        check_is_fitted(self, "model_", msg="Model is not fitted.")
        return self.model_.predict(self._validate_X(X))

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Returns model probability predictions.
        """
        check_is_fitted(self, "model_", msg="Model is not fitted.")
        return self.model_.predict_proba(self._validate_X(X))

    @property
    def scorer(self) -> tp.Callable[..., float]:
        """
        Returns correct scorer to use when scoring with RegressionCV.
        """
        return make_scorer(
            classification_metrics[self._metric], greater_is_better=True
        )
