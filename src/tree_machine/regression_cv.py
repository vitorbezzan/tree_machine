"""
Definition for RegressionCV.
"""

import multiprocessing
import typing as tp
from functools import partial

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import NonNegativeInt, validate_call
from pydantic.dataclasses import dataclass
from sklearn.base import RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from .base import BaseAutoCV
from .explainer import ExplainerMixIn
from .optimizer_params import BalancedParams, OptimizerParams
from .regression_metrics import AcceptableRegression, regression_metrics
from .types import GroundTruth, Inputs, Predictions

try:
    from shap import TreeExplainer
except ModuleNotFoundError:

    class TreeExplainer:  # type: ignore
        def __init__(self, **kwargs):
            raise RuntimeError("shap package is not available in your platform.")


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class RegressionCVConfig:
    """
    Available config to use when fitting a regression model.

    monotone_constraints: dictionary containing monotonicity direction allowed for each
        variable. 0 means no monotonicity, 1 means increasing and -1 means decreasing
        monotonicity.
    interactions: list of lists containing permitted relationships in data.
    n_jobs: Number of jobs to use when fitting the model.
    parameters: dictionary with distribution bounds for each hyperparameter to search
        on during optimization.
    return_train_score: whether to return the train score when fitting the model.
    quantile_alpha: Quantile alpha to use when fitting the model, if fitting a quantile
        model.
    """

    monotone_constraints: dict[str, int]
    interactions: list[list[str]]
    n_jobs: int
    parameters: OptimizerParams
    return_train_score: bool

    def get_kwargs(self, feature_names: list[str]) -> dict:
        """
        Returns parsed and validated constraint configuration for a RegressionCV model.

        Args:
            feature_names: list of feature names. If empty, will return empty
                constraints dictionaries and lists.
        """
        kwargs = {
            "monotone_constraints": {
                feature_names.index(key): value
                for key, value in self.monotone_constraints.items()
            },
            "interaction_constraints": [
                [feature_names.index(key) for key in lt] for lt in self.interactions
            ],
            "n_jobs": self.n_jobs,
        }

        return kwargs


default_regression = RegressionCVConfig(
    monotone_constraints={},
    interactions=[],
    n_jobs=multiprocessing.cpu_count() - 1,
    parameters=OptimizerParams(),
    return_train_score=True,
)


balanced_regression = RegressionCVConfig(
    monotone_constraints={},
    interactions=[],
    n_jobs=multiprocessing.cpu_count() - 1,
    parameters=BalancedParams(),
    return_train_score=True,
)


class RegressionCV(BaseAutoCV, RegressorMixin, ExplainerMixIn):
    """
    Defines an auto regression tree, based on the bayesian optimization base class.
    """

    model_: XGBRegressor
    feature_importances_: NDArray[np.float64]
    explainer_: TreeExplainer

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: AcceptableRegression,
        cv: BaseCrossValidator,
        n_trials: NonNegativeInt,
        timeout: NonNegativeInt,
        config: RegressionCVConfig,
    ) -> None:
        """
        Constructor for RegressionCV.

        Args:
            metric: Loss metric to use as base for the estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            config: Configuration to use when fitting the model.
        """
        super().__init__(metric, cv, n_trials, timeout)
        self.config = config

    def explain(self, X: Inputs, **explainer_params) -> dict[str, NDArray[np.float64]]:
        """
        Explains the inputs.
        """

        check_is_fitted(self, "model_", msg="Model is not fitted.")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = TreeExplainer(self.model_, **explainer_params)

        return {
            "mean_value": self.explainer_.expected_value,
            "shap_values": self.explainer_.shap_values(self._validate_X(X)),
        }

    def fit(self, X: Inputs, y: GroundTruth, **fit_params) -> "RegressionCV":
        """
        Fits RegressionCV.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
        """
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else []

        constraints = self.config.get_kwargs(self.feature_names_)

        self.model_ = self.optimize(
            estimator_type=partial(XGBRegressor, enable_categorical=True),
            X=self._validate_X(X),
            y=self._validate_y(y),
            parameters=self.config.parameters,
            return_train_score=self.config.return_train_score,
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

    @property
    def scorer(self) -> tp.Callable[..., float]:
        """
        Returns correct scorer to use when scoring with RegressionCV.
        """
        metric_func = self._resolve_metric(regression_metrics)
        return make_scorer(metric_func, greater_is_better=False)
