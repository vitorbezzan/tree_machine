"""
Definition of a auto classification tree.
"""
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from .base import BaseTree, SplitterLike
from .types import Actuals, Inputs, regression_metrics

default_grid = {
    "n_estimators": (2, 100),
    "num_leaves": (20, 200),
}


class RegressorTree(BaseTree, RegressorMixin):
    """
    Defines an auto regressor tree. Uses bayesian optimisation to select a set of
    hyperparameters automatically, and accepts user intervention over the parameters
    to be selected and their domains.
    """

    def __init__(
        self,
        metric: str = "f1",
        split: SplitterLike = KFold(n_splits=5),
        optimisation_iter: int = 32,
    ) -> None:
        """
        Constructor for RegressorTree.
        See BaseTree for more details.
        """
        super().__init__(
            "regression",
            metric,
            split,
            optimisation_iter,
        )

    def fit(self, X: Inputs, y: Actuals, **fit_params) -> "RegressorTree":
        """
        Fits estimator using bayesian optimization to select hyperparameters.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass for the
            internal solver:
                `grid`: dictionary containing the grid to be used in the optimisation
                  process.
                For all other parameters to pass to estimator, please append
                  "estimator__" to their name so the pipeline can route them directly to
                  the tree algorithm.
        """
        self._pre_fit(X)

        base_params = fit_params.pop("grid", default_grid)

        optimiser = self._create_optimiser(
            pipe=Pipeline(
                [
                    ("estimator", LGBMRegressor(n_jobs=-1, verbose=0)),
                ]
            ),
            params={f"estimator__{key}": base_params[key] for key in base_params},
            metric=regression_metrics.get(self.metric, "mse"),
        )

        optimiser.fit(self._treat_dataframe(X, self.feature_names), y, **fit_params)

        self.best_params = optimiser.best_params_
        self.model_ = optimiser.best_estimator_.steps[0][1]

        return self

    def score(self, X: Inputs, y: Actuals, sample_weight=None) -> float:
        """
        Returns model score. Sample weight is not used here, but kept for compatibility.
        Please use bootstrapping to add weights to samples.
        """
        return self._score(X, y, regression_metrics.get(self.metric, "mse"))
