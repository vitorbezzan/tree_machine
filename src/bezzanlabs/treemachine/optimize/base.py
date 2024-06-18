"""
Base class for optimizer using bayesian optimisation.
"""
import typing as tp
from dataclasses import dataclass

import pandas as pd
from optuna.distributions import BaseDistribution
from optuna.integration import OptunaSearchCV
from optuna.trial import FrozenTrial
from sklearn.model_selection import BaseCrossValidator, KFold

from ..types import Actuals, Inputs, Pipe


@dataclass(frozen=True)
class OptimizerConfig:
    """
    Configuration dictionary for `OptimiserEstimatorMixIn`.
    """

    n_trials: int
    timeout: int
    cv: BaseCrossValidator = KFold(n_splits=5)
    return_train_score: bool = True


class OptimizerEstimatorMixIn(object):
    """
    MixIn class to estimate models using Bayesian Optimisation.
    """

    optimizer_: OptunaSearchCV
    trials_: list[FrozenTrial]
    best_params_: dict[str, object]

    @property
    def optimized(self):
        """
        Checks if current instance have an optimised model.
        """
        return hasattr(self, "optimizer_")

    def _fit(
        self,
        estimator: Pipe,
        X: Inputs,
        y: Actuals,
        grid: dict[str, BaseDistribution],
        scorer: tp.Callable[..., float],
        optimiser_config: OptimizerConfig,
    ) -> "OptunaSearchCV":
        """
        Optimises estimator using Bayesian Optimisation.

        Args:
            estimator: Estimator to optimize hyperparameters.
            X: Data input for estimator.
            y: Target input for estimator.
            grid: Dictionary containing parameter name to optimize and respective
                distribution to use.
            scorer: Function to use as scoring for optimiser. Usually the result of
                `make_scorer` function from scikit-learn or similar.
            optimiser_config: Configuration to use for search procedure.
        """
        self.optimizer_ = OptunaSearchCV(
            estimator,
            cv=optimiser_config.cv,
            param_distributions=grid,
            scoring=scorer,
            n_trials=optimiser_config.n_trials,
            timeout=optimiser_config.timeout,
            return_train_score=optimiser_config.return_train_score,
        ).fit(X, y)

        self.trials_ = self.optimizer_.trials_
        self.best_params_ = self.optimizer_.best_params_

        return self.optimizer_

    def get_stats_df(self) -> pd.DataFrame:
        """
        Gets stats DataFrame.
        """
        return pd.DataFrame(
            [
                {
                    "train": trial.user_attrs["mean_train_score"],
                    "train_std": trial.user_attrs["std_train_score"],
                    "test": trial.user_attrs["mean_test_score"],
                    "test_std": trial.user_attrs["std_test_score"],
                }
                for i, trial in enumerate(self.trials_)
            ]
        ).sort_values(by="test", ascending=True)
