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
from sklearn.utils.validation import _check_y, check_array, check_is_fitted

from .optimizer_params import OptimizerParams
from .types import GroundTruth, Inputs, Predictions


class BaseAutoCV(ABC, BaseEstimator):
    """
    Defines BaseAutoCV, a class to help fit models using Bayesian optimization.
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
    ) -> None:
        """
        Constructor for BaseAutoTreeCV.

        Args:
            metric: Loss metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
        """
        self.metric = metric
        self.cv = cv
        self.n_trials = n_trials
        self.timeout = timeout

    def explain(self, X: Inputs, **explainer_params):
        """
        Explains the inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: Inputs) -> Predictions:
        """
        Abstract implementation for a prediction function.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Abstract implementation for a prediction function, returning probabilities.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def scorer(self) -> tp.Callable[..., float]:
        """
        Abstract implementation for a function that returns the correct scorer to use
        when fitting/scoring models.
        """
        raise NotImplementedError()

    @property
    def study(self) -> Study:
        check_is_fitted(self, "model_", msg="Model is not fitted.")
        return self.study_

    @property
    def cv_results(self) -> NDArray[np.float64]:
        """
        Returns test score for each fold for the model best estimator.
        """
        check_is_fitted(self, "model_", msg="Model is not fitted.")
        return self.study.best_trial.user_attrs["cv_results"]["test_score"]

    def optimize(
        self,
        estimator_type,
        X: Inputs,
        y: GroundTruth,
        parameters: OptimizerParams,
        return_train_score: bool,
        **kwargs,
    ):
        """
        Fits a model using Bayesian optimization and optuna.

        Args:
            estimator_type: type of object to use when fitting models.
            X: Input data to use when fitting models.
            y: Ground truth data to use when fitting models.
            return_train_score: Whether to return or not the training score for
                optimization.
            parameters: Distributions defined by user to select trial values.

        Returns:
            Fitted `estimator_type` object, using the best parameters selected using
              Bayesian optimization.
        """

        def _objective(trial: Trial) -> float:
            """Objective function to use in optimization."""
            estimator = estimator_type(
                **kwargs,
                **parameters.get_trial_values(trial),
            )

            cv_results = cross_validate(
                estimator,
                X,
                y,
                scoring=self.scorer,
                cv=self.cv,
                return_train_score=return_train_score,
            )
            trial.set_user_attr("cv_results", cv_results)
            return np.mean(cv_results["test_score"])

        self.study_ = create_study(
            direction="maximize",
            sampler=TPESampler(),
            pruner=HyperbandPruner(),
        )
        self.study_.optimize(_objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params_ = self.study_.best_params

        return estimator_type(**self.best_params_, **kwargs).fit(X, y)

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
