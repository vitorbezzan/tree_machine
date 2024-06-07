"""
BaseAuto tree class for AutoML trees.
"""
from abc import ABC

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shap import TreeExplainer
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_y, check_array, check_is_fitted
from xgboost import XGBModel

from bezzanlabs.treemachine.optimize import OptimizerEstimatorMixIn
from bezzanlabs.treemachine.splitter_proto import SplitterLike
from bezzanlabs.treemachine.types import Actuals, Inputs, Predictions


class BaseAuto(ABC, BaseEstimator, OptimizerEstimatorMixIn):
    """
    Defines a base behavior for all AutoTrees.
    """

    model_: XGBModel
    explainer_: TreeExplainer
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
        Constructor for BaseAuto.

        Args:
            task: Specifies which task this tree ensemble performs. Accepts "regression"
                or "classifier".
            metric: Metric to use as base for estimation process. Depends on "task".
            cv: Splitter object to use when estimating the model.
            optimisation_iter: Number of rounds to use in optimisation.
        """
        self.task = task
        self.metric = metric
        self.cv = cv
        self.optimisation_iter = optimisation_iter

        self.feature_names: list[str] = []

    def explain(self, X: Inputs, **explain_params) -> dict[str, object]:
        """
        Explains data using shap values.

        Returns:
            For regression a dictionary with keys:
                shap_values: np.array of shap values per input variable
                    (n_samples, n_vars).
                mean_value: float with mean value for target.
            For binary classification:
                shap_values: np.array of shap values per input variable
                    (n_samples, n_vars).
                mean_value: mean probability for positive class.
            For multiclass classification:
                shap_values: A list of np.array of shap values per input variable
                    (n_samples, n_vars) per class.
                mean_value: A list of mean probabilities for each class.
        """
        check_is_fitted(self, "model_", msg="Model is not fitted.")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = TreeExplainer(self.model_, **explain_params)

        return {
            "shap_values": self.explainer_.shap_values(self._treat_x(X)),
            "mean_value": self.explainer_.expected_value,
        }

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model predictions.
        """
        check_is_fitted(self, "model_")

        return self.model_.predict(self._treat_x(X))

    def _treat_x(
        self,
        X: Inputs,
    ) -> NDArray[np.float64]:
        """
        Checks if inputs are consistent and have the expected columns.
        """
        if isinstance(X, pd.DataFrame):
            return check_array(  # type: ignore
                np.array(X[self.feature_names or X.columns]),
            )

        return check_array(X)  # type: ignore

    @staticmethod
    def _treat_y(
        y: Actuals,
    ) -> NDArray[np.float64]:
        """
        Checks if Actual/Predictions are consistent and have the expected properties.
        """
        return _check_y(y, multi_output=False, y_numeric=True)
