"""
Base class to define auto trees.
"""
from abc import ABC

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shap import TreeExplainer
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_y, check_array, check_is_fitted
from xgboost import XGBModel

from ..optimize import OptimizerEstimatorMixIn
from ..splitter_proto import SplitterLike
from ..types import Actuals, Inputs, Predictions


class BaseAuto(ABC, BaseEstimator, OptimizerEstimatorMixIn):
    """
    Defines BaseAuto, base class for all auto trees.
    """

    model_: XGBModel
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
        """
        check_is_fitted(self, "model_", msg="Model is not fitted.")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = TreeExplainer(self.model_, **explain_params)

        explainer_shap = self.explainer_.shap_values(self._treat_x(X))
        if explainer_shap.ndim == 3:
            shap_values = [
                pd.DataFrame(explainer_shap[:, :, i], columns=self.feature_names)
                for i in range(explainer_shap.shape[2])
            ]
        else:
            shap_values = pd.DataFrame(  # type: ignore
                explainer_shap,
                columns=self.feature_names,
            )

        return {
            "shap_values": shap_values,
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
