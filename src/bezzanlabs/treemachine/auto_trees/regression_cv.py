"""
Regression auto tree.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import AfterValidator, NonNegativeInt, validate_call
from pydantic.types import Annotated
from sklearn.base import RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .base import BaseAutoTree
from .defaults import defaults
from .metrics import regression_metrics
from .types import Actuals, Inputs


def _is_regression_metric(metric: str) -> str:
    assert metric in regression_metrics
    return metric


AcceptableMetric = Annotated[str, AfterValidator(_is_regression_metric)]


class RegressionCV(BaseAutoTree, RegressorMixin):
    """
    Defines an auto regression tree.
    """

    model_: XGBRegressor
    feature_importances_: NDArray[np.float64]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        metric: AcceptableMetric = "mse",
        cv: BaseCrossValidator = KFold(n_splits=5),
        n_trials: NonNegativeInt = 100,
        timeout: NonNegativeInt = 180,
    ) -> None:
        """
        Constructor for RegressionCV.

        Args:
            metric: Metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
        """
        super().__init__(metric, cv, n_trials, timeout)

    def fit(self, X: Inputs, y: Actuals, **fit_params) -> "RegressionCV":
        """
        Fits estimator using bayesian optimization.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass to the base
                regression or the parameter distribution.
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

        pipeline = [
            (
                "estimator",
                XGBRegressor(
                    n_jobs=-1,
                    enable_categorical=True,
                    monotone_constraints=fit_params.get("monotone_constraints", None),
                    interaction_constraints=fit_params.get("interactions", None),
                ),
            ),
        ]

        distributions = {
            f"estimator__{k}": v
            for k, v in fit_params.get("distributions", defaults).items()
        }

        self._fit(
            Pipeline(pipeline),
            X,
            y,
            distributions,
            make_scorer(regression_metrics[self.metric], greater_is_better=True),
            self.cv,
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
        return -regression_metrics[self.metric](
            self._treat_y(y),
            self.predict(X),
            sample_weight=sample_weight,
        )
