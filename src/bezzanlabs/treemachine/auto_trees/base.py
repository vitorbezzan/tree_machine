"""
BaseAuto tree class for AutoML trees.
"""
from abc import ABC

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shap import TreeExplainer  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore
from sklearn.utils.validation import _check_y, check_is_fitted
from skopt import BayesSearchCV  # type: ignore
from xgboost import XGBModel

from ..types import Actuals, Inputs, Pipe, Predictions
from .fixes import apply_patches
from .splitter_proto import SplitterLike


class BaseAuto(ABC, BaseEstimator):
    """
    Defines a base, which encapsulates the basic behavior of all trees in the
    package.
    """

    model_: XGBModel
    explainer_: TreeExplainer
    best_params_: dict[str, object]
    cv_results_: dict[str, object]
    feature_importances_: NDArray[np.float64]

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
        cv: SplitterLike,
        optimisation_iter: int,
    ) -> None:
        """
        Constructor for BaseAuto (AutoTrees).

        Args:
            task: Specifies which task this tree ensemble performs. Suggestions are
            "regression" or "classifier".
            metric: Metric to use as base for estimation process. Depends on "task".
            cv: Splitter object to use when estimating the model.
            optimisation_iter: Number of rounds to use in optimisation.
        """
        self.task = task
        self.metric = metric
        self.cv = cv
        self.optimisation_iter = optimisation_iter

        self.feature_names: list[str] | None = None

    def explain(self, X: Inputs, **explain_params) -> dict[str, object]:
        """
        Explains data using shap values.

        Returns:
            For regression a dictionary with keys:
                shap_values: np.array of shap values per input variable
                (n_samples, n_vars)
                mean_value: float with mean value for target.
            For binary classification:
                shap_values: np.array of shap values per input variable
                (n_samples, n_vars)
                mean_value: mean probability for positive class.
            For multiclass classification:
                shap_values: A list of np.array of shap values per input variable
                (n_samples, n_vars) per class.
                mean_value: A list of mean probabilities for each class.
        """
        check_is_fitted(self, "model_", msg="Model is nt fitted.")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = TreeExplainer(self.model_, **explain_params)

        return {
            "shap_values": self.explainer_.shap_values(
                self._treat_x(X, self.feature_names)
            ),
            "mean_value": self.explainer_.expected_value,
        }

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model prediction. For regression returns the regression values, and for
        classification, there is an override that returns the class predictions.
        """
        check_is_fitted(self, "model_")

        return self.model_.predict(self._treat_x(X, self.feature_names))

    def _create_optimiser(
        self,
        pipe: Pipe,
        params: dict[str, object],
        metric: str,
    ) -> BayesSearchCV:
        return BayesSearchCV(
            pipe,
            params,
            n_iter=self.optimisation_iter,
            cv=self.cv,
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
    def _treat_x(
        X: Inputs,
        feature_names: list[str] | None = None,
    ) -> NDArray[np.float64]:
        if isinstance(X, pd.DataFrame):
            return check_array(X[feature_names or X.columns].values)  # type: ignore

        return check_array(X)  # type: ignore

    @staticmethod
    def _treat_y(
        y: Actuals,
    ) -> NDArray[np.float64]:
        return _check_y(y, multi_output=False)
