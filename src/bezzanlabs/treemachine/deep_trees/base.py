"""
Defines base class for all deep trees.
"""
from abc import ABC

import numpy as np
import pandas as pd
import shap  # type: ignore
from keras.models import Model  # type: ignore
from numpy.typing import NDArray
from shap import DeepExplainer  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore
from sklearn.utils.validation import check_is_fitted

from ..types import Actuals, Inputs


class BaseDeep(ABC, BaseEstimator):
    """
    Defines a base, which encapsulates the basic behavior of all deep trees in the
    package.
    """

    model_: Model
    explainer_: DeepExplainer

    def __new__(cls, *args, **kwargs):
        if cls is BaseDeep:
            raise TypeError(
                "BaseDeep is not directly instantiable.",
            )  # pragma: no cover
        return super(BaseDeep, cls).__new__(cls)

    def __init__(
        self,
        task: str,
        n_estimators: int,
        internal_size: int,
        max_depth: int,
        feature_fraction: float,
        explain_fraction: float,
    ) -> None:
        """
        Constructor for BaseAuto (DeepTrees).

        Args:
            task: Specifies which task this tree ensemble performs. Accepts "regression"
            or "classifier".
        """
        self.task = task
        self.feature_names: list[str] | None = None
        self.labeler: MultiLabelBinarizer = MultiLabelBinarizer()

        self.n_estimators = n_estimators
        self.internal_size = internal_size
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction
        self.explain_fraction = explain_fraction

    def explain(self, X: Inputs, **explain_params) -> tuple[NDArray[np.float64], float]:
        """
        Explains data using shap values. Beware that it uses KernelExplainer, which
        takes a long time to compute depending on the size of the dataset and the number
        of samples to be used.

        Please check the shap documentation for more information on the parameters the
        function accepts.

        Args:
            **explain_params: parameters to pass to shap KernelExplainer.

        Returns:
            array (or list of) with prediction explanations + mean value
        """
        check_is_fitted(self, "model_")
        check_is_fitted(self, "explainer_")

        shap.explainers._deep.deep_tf.op_handlers[
            "AddV2"
        ] = shap.explainers._deep.deep_tf.passthrough
        return (
            self.explainer_.shap_values(
                self._treat_dataframe(X, self.feature_names),
                **explain_params,
            ),
            self.explainer_.expected_value,
        )

    def _pre_fit(self, X: Inputs, y: Actuals) -> tuple[Inputs, Actuals]:
        """
        BaseAuto procedures for fitting models.

        - For regression, keeps the same shape and values, just adjusting the
        output_size.
        - For classification, changes the shape, and encodes the values.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        if self.task == "regression":
            return np.array(X), np.array(y).reshape(X.shape[0], -1)

        return np.array(X), self.labeler.fit_transform(
            np.reshape(np.array(y), newshape=(X.shape[0], -1)),
        )

    @staticmethod
    def _treat_dataframe(
        X: Inputs,
        feature_names: list[str] | None = None,
    ) -> Inputs:
        if isinstance(X, pd.DataFrame):
            return check_array(X[feature_names or X.columns].values)

        return check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
        )
