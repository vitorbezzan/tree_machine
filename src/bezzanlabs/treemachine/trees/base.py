"""
Base tree class.
"""
import typing as tp
from abc import ABC

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMModel
from numpy.typing import NDArray
from shap import Explainer, TreeExplainer  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore
from sklearn.utils.validation import check_is_fitted
from skopt import BayesSearchCV  # type: ignore

from .fixes import apply_patches
from .types import Actuals, Inputs, Pipe, Predictions


@tp.runtime_checkable
class SplitterLike(tp.Protocol):
    """
    Specifies a protocol for splitters. Defines the minimum specified behavior for these
    types of objects.
    """

    def get_n_splits(
        self,
        X: Inputs,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> int:
        """
        Returns the number of splits for a given dataset.
        """

    def split(
        self,
        X: Inputs,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> tp.Iterable[tp.Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Splits and yields data.
        """


class BaseTree(ABC, BaseEstimator):
    """
    Defines a BaseTree, which encapsulates the basic behavior of all trees in the
    package.
    """

    best_params: dict[str, tp.Any]
    model_: LGBMModel
    explainer_: Explainer | TreeExplainer

    def __new__(cls, *args, **kwargs):
        if cls is BaseTree:
            raise TypeError(
                "BaseTree is not directly instantiable.",
            )  # pragma: no cover
        return super(BaseTree, cls).__new__(cls)

    def __init__(
        self, task: str, metric: str, split: SplitterLike, optimisation_iter: int
    ):
        """
        Constructor for BaseTree.

        Args:
            task: Specifies which task this tree ensemble performs. Suggestions are
            "regression" or "classifier".
            metric: Metric to use as base for estimation process. Depends on "task".
            split: Splitter object to use when estimating the model.
            optimisation_iter: Number of rounds to use in optimisation.
        """
        self.task = task
        self.metric = metric
        self.split = split
        self.optimisation_iter = optimisation_iter

        self.feature_names: list[str] | None = None

    @property
    def best_params_(self) -> dict[str, tp.Any] | None:
        return getattr(self, "best_params_", None)  # pragma: no cover

    @property
    def feature_importances_(self) -> NDArray[np.float64] | dict[str, float] | None:
        """
        Returns feature importances from selected model.
        """
        check_is_fitted(self, "model_")

        if self.feature_names is None:
            return self.model_.feature_importances_

        return dict(zip(self.feature_names, self.model_.feature_importances_))

    def _pre_fit(self, X: Inputs) -> None:
        """
        Base procedures for fitting models.
        """
        apply_patches()
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model prediction. For regression returns the regression values, and for
        classification, returns the class.
        """
        check_is_fitted(self, "model_")

        return self.model_.predict(
            self._treat_dataframe(X, self.feature_names),
        ).reshape(X.shape[0], -1)

    def explain(self, X: Inputs) -> tuple[NDArray[np.float64] | pd.DataFrame, float]:
        """
        Explains data using shap values.

        Returns:
            array with prediction explanations + mean value
        """
        check_is_fitted(self, "model_")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = Explainer(self.model_)

        return (
            self.explainer_(self._treat_dataframe(X, self.feature_names)).values,
            self.explainer_.expected_value,
        )

    @staticmethod
    def _treat_dataframe(
        X: Inputs,
        feature_names: list[str] | None = None,
    ):
        if isinstance(X, pd.DataFrame):
            return check_array(X[feature_names or X.columns].values)

        return check_array(X)

    def _create_optimiser(
        self,
        pipe: Pipe,
        params: dict[str, tp.Any],
        metric: str,
    ) -> BayesSearchCV:
        return BayesSearchCV(
            pipe,
            params,
            n_iter=self.optimisation_iter,
            cv=self.split,
            scoring=metric,
            verbose=False,
        )

    def _score(self, X: Inputs, y: Actuals, metric) -> float:
        check_is_fitted(self, "model_")
        return metric(self.model_, self._treat_dataframe(X, self.feature_names), y)
