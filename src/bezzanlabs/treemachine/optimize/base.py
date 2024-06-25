"""
Base class for optimizer using bayesian optimisation.
"""
import typing as tp

import pandas as pd
from optuna.distributions import BaseDistribution
from optuna.integration import OptunaSearchCV
from pydantic import NonNegativeInt, validate_call
from sklearn.model_selection import BaseCrossValidator

from ..types import Actuals, Inputs, Pipe


class OptimizerCVMixIn:
    """
    Adds optimizer capability to an estimator object.
    """

    n_trials_: int
    timeout_: int
    return_train_score_: bool

    optimizer_: OptunaSearchCV

    @validate_call
    def setup(
        self,
        n_trials: NonNegativeInt,
        timeout: NonNegativeInt,
        return_train_score: bool,
    ):
        """
        Sets the configuration for the optimizer.

        Args:
            n_trials: Number of trials to use in optimization.
            timeout: Timeout in seconds.
            return_train_score: Boll indicating if training scores should be returned.
        """
        self.n_trials_ = n_trials
        self.timeout_ = timeout
        self.return_train_score_ = return_train_score

    @property
    def is_setup(self) -> bool:
        return hasattr(self, "n_trials_")

    @property
    def optimized(self) -> bool:
        return hasattr(self, "optimizer_")

    @property
    def cv_results_(self) -> pd.DataFrame:
        if self.optimized:
            return pd.DataFrame(
                [
                    {
                        "train": trial.user_attrs["mean_train_score"],
                        "train_std": trial.user_attrs["std_train_score"],
                        "test": trial.user_attrs["mean_test_score"],
                        "test_std": trial.user_attrs["std_test_score"],
                    }
                    for i, trial in enumerate(self.optimizer_.trials_)
                ]
            ).sort_values(by="test", ascending=True)

        raise RuntimeError("Optimizer not fitted.")

    def _fit(
        self,
        estimator: Pipe,
        X: Inputs,
        y: Actuals,
        grid: dict[str, BaseDistribution],
        scorer: tp.Callable[..., float],
        cv: BaseCrossValidator,
    ) -> "OptunaSearchCV":
        if self.is_setup:
            self.optimizer_ = OptunaSearchCV(
                estimator,
                cv=cv,
                param_distributions=grid,
                scoring=scorer,
                n_trials=self.n_trials_,
                timeout=self.timeout_,
                return_train_score=self.return_train_score_,
            ).fit(X, y)

            return self.optimizer_

        raise RuntimeError("Optimizer not configured.")
