"""
Definition for RegressionCV.
"""
import typing as tp

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import NonNegativeInt, validate_call
from shap import TreeExplainer
from sklearn.base import RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from .base import BaseAutoCV, TExplainerResultsBase
from .parameters import TUserDistributionBase
from .regression_metrics import AcceptableRegression, regression_metrics
from .types import GroundTruth, Inputs, Predictions


class RegressionCVConfig(tp.TypedDict, total=False):
    """
    Available config to use when fitting a regression model.

    monotone_constraints: dictionary containing monotonicity direction allowed for each
        variable. 0 means no monotonicity, 1 means increasing and -1 means decreasing
        monotonicity.
    interactions: list of lists containing permitted relationships in data.
    parameters: dictionary with distribution bounds for each hyperparameter to search
        on during optimization.
    """

    monotone_constraints: dict[str, int]
    interactions: list[list[str]]
    parameters: TUserDistributionBase


default_regression = RegressionCVConfig(
    monotone_constraints={},
    interactions=[],
    parameters=TUserDistributionBase(),
)


class RegressionCV(BaseAutoCV, RegressorMixin):
    """
    Defines an auto regression tree.
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
        n_jobs: int = -1,
    ) -> None:
        """
        Constructor for RegressionCV.

        Args:
            metric: Loss metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            n_jobs: Number of processes to use internally when estimating the model.
        """
        super().__init__(metric, cv, n_trials, timeout, n_jobs)
        self._config = config

    def fit(self, X: Inputs, y: GroundTruth, **fit_params) -> "RegressionCV":
        """
        Fits RegressionCV, using Bayesian optimization and optuna backend.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)

            monotone_constraints = {
                self.feature_names_.index(key): value
                for key, value in self._config["monotone_constraints"].items()
            }
            interaction_constraints = [
                [self.feature_names_.index(key) for key in lt]
                for lt in self._config["interactions"]
            ]
        else:
            monotone_constraints = self._config["monotone_constraints"]  # type: ignore
            interaction_constraints = self._config["interactions"]  # type: ignore

        self.model_ = self.optimize(
            estimator_type=XGBRegressor,
            X=self._validate_X(X),
            y=self._validate_y(y),
            parameters=fit_params.get("parameters", TUserDistributionBase()),
            scorer=make_scorer(
                regression_metrics[self._metric], greater_is_better=False
            ),
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
        )
        self.feature_importances_ = self.model_.feature_importances_

        return self

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model predictions.

        For a regression model, returns a real value. For a classifier, outputs the
        predicted class of the object.
        """
        check_is_fitted(self, "model_", msg="Model is not fitted.")
        return self.model_.predict(self._validate_X(X))

    @validate_call(config={"arbitrary_types_allowed": True})
    def explain(self, X: Inputs, **explainer_params) -> TExplainerResultsBase:
        check_is_fitted(self, "model_", msg="Model is not fitted.")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = TreeExplainer(self.model_, **explainer_params)

        return TExplainerResultsBase(
            mean_value=tp.cast(
                float,
                self.explainer_.expected_value,
            ),
            shap_values=tp.cast(
                NDArray[np.float64],
                self.explainer_.shap_values(self._validate_X(X)),
            ),
        )

    def score(
        self,
        X: Inputs,
        y: GroundTruth,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Returns model score.
        """
        return -regression_metrics[self._metric](
            self._validate_y(y),
            self.predict(X),
            sample_weight=sample_weight,
        )
