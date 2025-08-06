"""
Definition for RegressionCV.
"""

import multiprocessing
import typing as tp
from functools import partial, update_wrapper

from pydantic.dataclasses import dataclass
from sklearn.metrics import make_scorer

from .optimizer_params import BalancedParams
from .regression_cv import RegressionCV, RegressionCVConfig
from .regression_metrics import regression_metrics


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class QuantileCVConfig(RegressionCVConfig):
    """
    Available config to use when fitting a quantile model.

    quantile_alpha: Quantile alpha to use when fitting the model.
    """

    quantile_alpha: float | None = None


def balanced_quantile(alpha: float) -> QuantileCVConfig:
    """Returns a Balanced regression CV config."""
    return QuantileCVConfig(
        monotone_constraints={},
        interactions=[],
        n_jobs=multiprocessing.cpu_count() - 1,
        parameters=BalancedParams(),
        return_train_score=True,
        quantile_alpha=alpha,
    )


class QuantileCV(RegressionCV):
    """
    Defines an auto regression tree, based on the bayesian optimization base class.
    """

    config: QuantileCVConfig

    @property
    def scorer(self) -> tp.Callable[..., float]:
        """
        Returns correct scorer to use when scoring with RegressionCV.
        """
        return make_scorer(
            update_wrapper(
                partial(
                    regression_metrics["quantile"],
                    alpha=self.config.quantile_alpha,
                ),
                regression_metrics["quantile"],
            ),
            greater_is_better=False,
        )
