# isort: skip_file
"""
Definition for ClassifierCV.
"""
import typing as tp

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from numpy.typing import NDArray
from pydantic import NonNegativeInt, validate_call
from pydantic.dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from imblearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from .base import BaseAutoCV
from .classification_metrics import AcceptableClassifier, classification_metrics
from .optimizer_params import OptimizerParams
from .types import GroundTruth, Inputs, Predictions


class _Identity(TransformerMixin):
    """Identity transformation."""

    def __init__(self):
        pass

    def fit(self, X: Inputs, y: GroundTruth) -> "_Identity":
        return self

    def transform(self, X: Inputs, y=None) -> Inputs:
        return X


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

    sampler: BaseSampler | None
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
            "sampler": self.sampler if self.sampler is not None else _Identity(),
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
    sampler=None,
    monotone_constraints={},
    interactions=[],
    n_jobs=-1,
    parameters=OptimizerParams(),
)


class _EstimatorWithSampler(BaseEstimator):
    """Helper class to ensure classifiers can be used with optimizers."""

    pipeline_: Pipeline

    def __init__(
        self, sampler, monotone_constraints, interaction_constraints, n_jobs, **kwargs
    ):
        self.sampler = sampler
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X, y):
        self.pipeline_ = Pipeline(
            [
                ("sampler", self.sampler),
                (
                    "estimator",
                    XGBClassifier(
                        monotone_constraints=self.monotone_constraints,
                        interaction_constraints=self.interaction_constraints,
                        n_jobs=self.n_jobs,
                        **self.kwargs,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)

    @property
    def fitted_estimator(self):
        return self.pipeline_.steps[1][1]


class ClassifierCV(BaseAutoCV, RegressorMixin):
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
            estimator_type=_EstimatorWithSampler,
            X=self._validate_X(X),
            y=self._validate_y(y),
            parameters=self._config.parameters,
            **constraints,
        ).fitted_estimator
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
