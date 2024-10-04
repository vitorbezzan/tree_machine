"""
Definition of BaseAutoCV.
"""
import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from optuna import Study, Trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from pydantic import NonNegativeInt, validate_call
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.utils.validation import _check_y, check_array

from .parameters import TUserDistributionBase
from .types import GroundTruth, Inputs, Predictions


class TExplainerResultsBase(tp.TypedDict, total=False):
    """
    Describes an expected result coming from an .explain() call.

    mean_value: If regressor/two-class classifier, returns the average value expected by
    the model. If multi-class classifier, returns array with expected values for
    each class.
    shap_values: If regressor/two-class classifier, returns array with shap
    contributions for each row and each variable. If multi-class classifier, returns
    list of arrays with expected shap contributions for each class, row and
    variables.
    """

    mean_value: float | NDArray[np.float64]
    shap_values: NDArray[np.float64] | list[NDArray[np.float64]]


class BaseAutoCV(ABC, BaseEstimator):
    """
    Defines BaseAuto, base class for all auto trees.
    """

    feature_names_: list[str]
    study_: Study
    best_params_: dict[str, float | int]

    def __new__(cls, *args, **kwargs):
        if cls is BaseAutoCV:
            raise TypeError(
                "BaseAutoCV is not directly instantiable.",
            )  # pragma: no cover
        return super(BaseAutoCV, cls).__new__(cls)

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: str,
        cv: BaseCrossValidator,
        n_trials: NonNegativeInt,
        timeout: NonNegativeInt,
        n_jobs: int,
    ) -> None:
        """
        Constructor for BaseAutoTreeCV.

        Args:
            metric: Loss metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            n_jobs: Number of processes to use internally when estimating the model.
        """
        self._metric = metric
        self._cv = cv
        self._n_trials = n_trials
        self._timeout = timeout
        self._n_jobs = n_jobs

    def optimize(
        self,
        estimator_type,
        X: Inputs,
        y: GroundTruth,
        parameters: TUserDistributionBase,
        scorer: tp.Callable[..., float],
        **kwargs,
    ):
        """
        Fits a model using Bayesian optimization and optuna.

        Args:
            estimator_type: type of object to use when fitting models.
            X: Input data to use when fitting models.
            y: Ground truth data to use when fitting models.
            parameters: Distributions defined by user to select trial values.
            scorer: Scorer function to use when selecting best models

        Returns:
            Fitted `estimator_type` object, using the best parameters selected using
              Bayesian optimization.
        """

        def _objective(trial: Trial) -> float:
            """Objective function to use in optimization."""
            estimator = estimator_type(
                n_jobs=self._n_jobs,
                **parameters.get_trial_values(trial),
                **kwargs,
            )

            cv_results = cross_validate(estimator, X, y, scoring=scorer, cv=self._cv)
            trial.set_user_attr("cv_results", cv_results)
            return np.mean(cv_results["test_score"])

        self.study_ = create_study(
            direction="maximize", sampler=TPESampler(), pruner=HyperbandPruner()
        )
        self.study_.optimize(_objective, n_trials=self._n_trials, timeout=self._timeout)
        self.best_params_ = self.study_.best_params

        return estimator_type(n_jobs=self._n_jobs, **self.best_params_, **kwargs).fit(
            X, y
        )

    @abstractmethod
    def predict(self, X: Inputs) -> Predictions:
        """
        Abstract implementation for a prediction function.
        """
        raise NotImplementedError()

    @abstractmethod
    def explain(self, X: Inputs, **explainer_params) -> TExplainerResultsBase:
        """
        Abstract implementation for a explainer function.
        """
        raise NotImplementedError()

    @validate_call(config={"arbitrary_types_allowed": True})
    def _validate_X(self, X: Inputs) -> Inputs:
        """
        Validates X and returns np.ndarray internal representation.
        """
        if isinstance(X, pd.DataFrame):
            return check_array(  # type: ignore
                np.array(X[getattr(self, "feature_names_", []) or X.columns]),
                dtype="numeric",
                force_all_finite="allow-nan",
            )

        return check_array(  # type: ignore
            np.array(X),
            dtype="numeric",
            force_all_finite="allow-nan",
        )

    @staticmethod
    @validate_call(config={"arbitrary_types_allowed": True})
    def _validate_y(y: GroundTruth) -> GroundTruth:
        """
        Validates y and returns np.ndarray internal representation.
        """
        return _check_y(np.array(y), multi_output=False, y_numeric=True)
