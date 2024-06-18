"""
Regressor auto tree.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ..optimize import OptimizerConfig
from ..splitter_proto import SplitterLike
from ..types import Actuals, Inputs
from .base import BaseAuto
from .config import defaults, regression_metrics


class Regressor(BaseAuto, RegressorMixin):
    """
    Defines an auto regressor tree. Uses bayesian optimisation to select a set of
    hyperparameters automatically, and accepts user intervention over the parameters
    to be selected and their domains.
    """

    model_: XGBRegressor
    feature_importances_: NDArray[np.float64]

    def __init__(
        self,
        metric: str = "mse",
        cv: SplitterLike = KFold(n_splits=5),
        optimisation_iter: int = 100,
    ) -> None:
        """
        Constructor for RegressorTree.
        """
        super().__init__(
            "regression",
            metric,
            cv,
            optimisation_iter,
        )

    def fit(self, X: Inputs, y: Actuals, **fit_params) -> "Regressor":
        """
        Fits estimator using bayesian optimization to select hyperparameters.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass for the
            internal solver.
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

        self._fit(
            Pipeline(
                [
                    (
                        "estimator",
                        XGBRegressor(
                            n_jobs=-1,
                            enable_categorical=True,
                            monotone_constraints=fit_params.get(
                                "monotone_constraints", None
                            ),
                            interaction_constraints=fit_params.get(
                                "interaction_constraints", None
                            ),
                        ),
                    )
                ]
            ),
            X,
            y,
            {
                f"estimator__{key}": value
                for key, value in fit_params.get("distributions", defaults).items()
            },
            make_scorer(regression_metrics[self.metric], greater_is_better=False),
            OptimizerConfig(
                n_trials=self.optimisation_iter,
                timeout=fit_params.get("timeout", 180),
                cv=self.cv,
                return_train_score=True,
            ),
        )

        self.model_ = self.optimizer_.best_estimator_.steps[0][1]
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
