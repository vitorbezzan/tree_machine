"""
Definition for RegressionCV.
"""

import typing as tp
from functools import partial, update_wrapper

from pydantic import NonNegativeFloat, NonNegativeInt, validate_call
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator

from .regression_cv import RegressionCV, RegressionCVConfig
from .regression_metrics import regression_metrics


class QuantileCV(RegressionCV):
    """
    Defines an auto regression tree, based on the bayesian optimization base class.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        alpha: NonNegativeFloat,
        cv: BaseCrossValidator,
        n_trials: NonNegativeInt,
        timeout: NonNegativeInt,
        config: RegressionCVConfig,
    ) -> None:
        """
        Constructor for RegressionCV.

        Args:
            alpha: The quantile to estimate, which must be between 0 and 1.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            config: Configuration to use when fitting the model.
        """
        super().__init__("quantile", cv, n_trials, timeout, config)
        self.alpha_ = alpha

    @property
    def scorer(self) -> tp.Callable[..., float]:
        """
        Returns correct scorer to use when scoring with RegressionCV.
        """
        return make_scorer(
            update_wrapper(
                partial(
                    regression_metrics["quantile"],
                    alpha=self.alpha_,
                ),
                regression_metrics["quantile"],
            ),
            greater_is_better=False,
        )
