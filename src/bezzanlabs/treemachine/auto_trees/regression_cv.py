"""
Regression auto tree.
"""
import typing as tp

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import NonNegativeInt, validate_call
from sklearn.base import RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .base import BaseAutoTree
from .defaults import TUsrDistribution, defaults, get_param_distributions
from .metrics import AcceptableRegression, regression_metrics
from .types import Actuals, Inputs


class RegressionCVOptions(tp.TypedDict, total=False):
    """
    Available options to use when fitting a regression model.

    monotone_constraints: dictionary containing monotonicity direction allowed for each
        variable. 0 means no monotonicity, 1 means increasing and -1 means decreasing
        monotonicity.
    interactions: list of lists containing permitted relationships in data.
    distributions: dictionary with distribution bounds for each hyperparameter to search
        on during optimization.
    """

    monotone_constraints: dict[str, int] | dict[int, int]
    interactions: list[list[int] | list[str]]
    distributions: TUsrDistribution


class RegressionCV(BaseAutoTree, RegressorMixin):
    """
    Defines an auto regression tree.
    """

    model_: XGBRegressor
    feature_importances_: NDArray[np.float64]

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: AcceptableRegression = "mse",
        cv: BaseCrossValidator = KFold(n_splits=5),
        n_trials: NonNegativeInt = 100,
        timeout: NonNegativeInt = 180,
        n_jobs: int = -1,
    ) -> None:
        """
        Constructor for RegressionCV.

        Args:
            metric: Metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            n_jobs: Number of processes to use internally when estimating the model.
        """
        super().__init__(metric, cv, n_trials, timeout, n_jobs)

    def fit(
        self,
        X: Inputs,
        y: Actuals,
        **fit_params: RegressionCVOptions,
    ) -> "RegressionCV":
        """
        Fits estimator using bayesian optimization.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass to the base
                regression or the parameter distribution.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        pipeline = [
            (
                "estimator",
                XGBRegressor(
                    n_jobs=self._n_jobs,
                    monotone_constraints=fit_params.get("monotone_constraints", None),
                    interaction_constraints=fit_params.get("interactions", None),
                ),
            ),
        ]

        distributions = get_param_distributions(
            tp.cast(TUsrDistribution, fit_params.get("distributions", defaults)),
        )

        self._fit(
            Pipeline(pipeline),
            self._treat_x(X),
            self._treat_y(y),
            {f"estimator__{k}": v for k, v in distributions.items()},
            make_scorer(regression_metrics[self._metric], greater_is_better=True),
            self._cv,
        )

        self.model_ = self.optimizer_.best_estimator_.steps[-1][1]
        self.feature_importances_ = self.model_.feature_importances_

        return self

    def score(
        self,
        X: Inputs,
        y: Actuals,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Returns model score.
        """
        return -regression_metrics[self._metric](
            self._treat_y(y),
            self.predict(X),
            sample_weight=sample_weight,
        )
