"""
Definition of a auto classification tree.
"""
import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin  # type: ignore
from sklearn.metrics import make_scorer  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from xgboost import XGBRegressor

from ..types import Actuals, Inputs
from .base import BaseAuto
from .config import default_hyperparams, regression_metrics
from .splitter_proto import SplitterLike


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
        optimisation_iter: int = 32,
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
        self._pre_fit(X)

        base_params = fit_params.pop("hyperparams", default_hyperparams)
        optimiser = self._create_optimiser(
            pipe=Pipeline(
                [
                    ("estimator", XGBRegressor(n_jobs=-1, classifier=0)),
                ]
            ),
            params={f"estimator__{key}": base_params[key] for key in base_params},
            metric=make_scorer(
                regression_metrics.get(self.metric, "mse"),
                greater_is_better=False,
            ),
        )

        optimiser.fit(
            self._treat_x(X, self.feature_names),
            self._treat_y(y),
            **fit_params,
        )

        self.model_ = optimiser.best_estimator_.steps[0][1]
        self.best_params_ = optimiser.best_params_
        self.cv_results_ = optimiser.cv_results_
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

        For regressors, returns (-1) * actual score since bigger is not better in this
        task.
        """
        return -regression_metrics.get(self.metric, "mse")(
            self._treat_y(y),
            self.predict(X),
            sample_weight=sample_weight,
        )
