"""
BaseAuto tree class for AutoML trees.
"""
import typing as tp
from abc import ABC

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMModel
from numpy.typing import NDArray
from shap import TreeExplainer  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore
from sklearn.utils.validation import check_is_fitted
from skopt import BayesSearchCV  # type: ignore

from ..types import Inputs, Pipe, Predictions
from .fixes import apply_patches
from .splitter_proto import SplitterLike


class BaseAuto(ABC, BaseEstimator):
    """
    Defines a base, which encapsulates the basic behavior of all trees in the
    package.
    """

    best_params: dict[str, tp.Any]
    model_: LGBMModel
    explainer_: TreeExplainer

    def __new__(cls, *args, **kwargs):
        if cls is BaseAuto:
            raise TypeError(
                "BaseAuto is not directly instantiable.",
            )  # pragma: no cover
        return super(BaseAuto, cls).__new__(cls)

    def __init__(
        self,
        task: str,
        metric: str,
        split: SplitterLike,
        optimisation_iter: int,
    ) -> None:
        """
        Constructor for BaseAuto (AutoTrees).

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
        Returns feature importance from selected model.
        """
        check_is_fitted(self, "model_")

        if self.feature_names is None:
            return self.model_.feature_importances_

        return dict(zip(self.feature_names, self.model_.feature_importances_))

    def explain(self, X: Inputs, **explain_params) -> tuple[NDArray[np.float64], float]:
        """
        Explains data using shap values.

        Returns:
            array with prediction explanations + mean value
        """
        check_is_fitted(self, "model_")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = TreeExplainer(self.model_)

        return (
            self.explainer_(self._treat_dataframe(X, self.feature_names)).values,
            self.explainer_.expected_value,
        )

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model prediction. For regression returns the regression values, and for
        classification, there is an override that returns the class predictions.
        """
        check_is_fitted(self, "model_")

        return self.model_.predict(self._treat_dataframe(X, self.feature_names))

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

    def _pre_fit(self, X: Inputs) -> None:
        """
        BaseAuto procedures for fitting models.
        """
        apply_patches()
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

    @staticmethod
    def _treat_dataframe(
        X: Inputs,
        feature_names: list[str] | None = None,
    ) -> Inputs:
        if isinstance(X, pd.DataFrame):
            return check_array(X[feature_names or X.columns].values)

        return check_array(X)
