# isort: skip_file
"""
Base class to define auto trees.
"""
import typing as tp
import numpy as np
import pandas as pd
from pydantic import NonNegativeInt, validate_call
from shap import TreeExplainer
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _check_y, check_array, check_is_fitted
from xgboost import XGBModel
from numpy.typing import NDArray

from .optimizer_base import OptimizerCVMixIn
from .types import Actuals, Inputs, Predictions


class TExplainerResults(tp.TypedDict, total=False):
    """
    Describes an expected result coming from an .explain() call.

    mean_value: If regressor/two-class classifier, returns the average value expected by
        the model. If multi-class classifier, returns array with expected values for
        each class.
    shap_values: If regressor/two-class classifier, returns array with shap
        contributions for each row and each variable. If multi-class classifier, returns
        list of arrays with expected shap contributions for each class, row and
        variables.
    """

    mean_value: float | NDArray[np.float64]
    shap_values: NDArray[np.float64] | list[NDArray[np.float64]]


class BaseAutoTree(BaseEstimator, OptimizerCVMixIn):
    """
    Defines BaseAuto, base class for all auto trees.
    """

    model_: XGBModel
    explainer_: TreeExplainer

    def __new__(cls, *args, **kwargs):
        if cls is BaseAutoTree:
            raise TypeError(
                "BaseAutoTree is not directly instantiable.",
            )  # pragma: no cover
        return super(BaseAutoTree, cls).__new__(cls)

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: str,
        cv: BaseCrossValidator,
        n_trials: NonNegativeInt = 100,
        timeout: NonNegativeInt = 180,
        n_jobs: int = -1,
    ) -> None:
        """
        Constructor for BaseAutoTree.

        Args:
            metric: Loss metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            n_jobs: Number of processes to use internally when estimating the model.
        """
        self._metric = metric
        self._cv = cv
        self._feature_names: list[str] = []
        self._n_jobs = n_jobs

        # Setup CVMixIn
        self.setup(n_trials, timeout, True)

    @validate_call(config={"arbitrary_types_allowed": True})
    def explain(self, X: Inputs, **explainer_params) -> TExplainerResults:
        """
        Explains data using shap values.

        Returns:
            TExplainerResults object with explanations.
        """
        check_is_fitted(self, "model_", msg="Model is not fitted.")

        if getattr(self, "explainer_", None) is None:
            self.explainer_ = TreeExplainer(self.model_, **explainer_params)

        explainer_shap = self.explainer_.shap_values(self._treat_x(X))

        if explainer_shap.ndim == 3:  # multi-class case
            return TExplainerResults(
                mean_value=tp.cast(
                    NDArray[np.float64],
                    self.explainer_.expected_value,
                ),
                shap_values=tp.cast(
                    list[NDArray[np.float64]],
                    [
                        np.array(explainer_shap[:, :, i])
                        for i in range(explainer_shap.shape[-1])
                    ],
                ),
            )
        else:  # regression/two-class case
            return TExplainerResults(
                mean_value=tp.cast(
                    float,
                    self.explainer_.expected_value,
                ),
                shap_values=tp.cast(
                    NDArray[np.float64],
                    explainer_shap,
                ),
            )

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model predictions.
        """
        check_is_fitted(self, "model_")
        return self.model_.predict(self._treat_x(X))

    @validate_call(config={"arbitrary_types_allowed": True})
    def _treat_x(self, X: Inputs) -> Inputs:
        """
        Checks and treats X inputs for model consumption.
        """
        if isinstance(X, pd.DataFrame):
            check_array(  # type: ignore
                np.array(X[self._feature_names or X.columns]),
                dtype="numeric",
                force_all_finite="allow-nan",
            )

            return X[self._feature_names or X.columns]

        return check_array(  # type: ignore
            np.array(X),
            dtype="numeric",
            force_all_finite="allow-nan",
        )

    @staticmethod
    @validate_call(config={"arbitrary_types_allowed": True})
    def _treat_y(y: Actuals) -> Actuals:
        """
        Checks and treats y inputs for model consumption.
        """
        return _check_y(np.array(y), multi_output=False, y_numeric=True)
