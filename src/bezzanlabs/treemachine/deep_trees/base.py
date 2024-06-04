"""
Defines base class for all deep trees.
"""
from abc import ABC

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import _check_y, check_array
from tensorflow.keras import Model

from bezzanlabs.treemachine.types import Actuals, Inputs


class BaseDeep(ABC, BaseEstimator):
    """
    Defines a base, which encapsulates the basic behavior of all deep trees in the
    package.
    """

    model_: Model

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
        alpha_l1: float = 0.0,
        lambda_l2: float = 0.0,
    ) -> None:
        """
        Constructor for BaseAuto (DeepTrees).

        Args:
            task: Specifies which task this tree ensemble performs. Accepts "regression"
                or "classifier".
        """
        self.task = task
        self.feature_names: list[str] = []
        self.labeler: MultiLabelBinarizer = MultiLabelBinarizer()

        self.n_estimators = n_estimators
        self.internal_size = internal_size
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction
        self.alpha_l1 = alpha_l1
        self.lambda_l2 = lambda_l2

    def _treat_x(
        self,
        X: Inputs,
    ) -> Inputs:
        """
        Checks if inputs are consistent and have the expected columns.
        """
        if isinstance(X, pd.DataFrame):
            return check_array(
                np.array(X[self.feature_names or X.columns]),
            )

        return check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
        )

    @staticmethod
    def _treat_y(
        y: Actuals,
    ) -> NDArray[np.float64]:
        """
        Checks if Actual/Predictions are consistent and have the expected properties.
        """
        return _check_y(y, multi_output=False, y_numeric=True)
