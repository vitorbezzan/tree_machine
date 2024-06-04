"""
Definition of a auto classification tree.
"""
import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from bezzanlabs.treemachine.splitter_proto import SplitterLike
from bezzanlabs.treemachine.types import Actuals, Inputs

from .base import BaseAuto
from .config import regression_metrics


class Regressor(BaseAuto, RegressorMixin):
    """
    Defines an auto regressor tree. Uses bayesian optimisation to select a set of
    hyperparameters automatically, and accepts user intervention over the parameters
    to be selected and their domains.
    """

    def __init__(
        self,
        metric: str = "mse",
        cv: SplitterLike = KFold(n_splits=5),
        optimisation_iter: int = 100,
    ) -> None:
        """
        Constructor for RegressorTree.
        See BaseTree for more details.
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
            internal solver:
                `hyperparams`: dictionary containing the space to be used in the
                optimisation process.

                For all other parameters to pass to estimator, please append
                "estimator__" to their name so the pipeline can route them directly to
                the tree algorithm. If using inside another pipeline, it need to be
                appended by an extra __.
        """
        optimised = self._fit(
            Pipeline([("estimator", XGBRegressor(n_jobs=-1))]),
            X,
            y,
            **fit_params,
        )

        self.model_ = optimised.best_estimator_.steps[0][1]
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
        return -regression_metrics.get(self.metric, "mse")(
            self._treat_y(y),
            self.predict(X),
            sample_weight=sample_weight,
        )
