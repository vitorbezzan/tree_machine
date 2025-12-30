"""Base types and utilities for Keras-backed deep-tree estimators."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pydantic import NonNegativeFloat, NonNegativeInt, validate_call
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from tensorflow.keras import Model

from tree_machine.types import GroundTruth, Inputs


class BaseDeep(ABC, BaseEstimator):
    """Shared base class for Keras-backed deep-tree estimators.

    This base:
        * stores common hyperparameters;
        * tracks feature names seen during fitting;
        * provides ``_validate_X`` to convert inputs to a numeric, finite numpy array.

    Subclasses must implement ``_validate_y``.
    """

    model_: Model

    def __new__(cls, *args, **kwargs):
        """Prevent direct instantiation of :class:`BaseDeep`."""
        if cls is BaseDeep:
            raise TypeError("BaseDeep is not directly instantiable.")
        return super(BaseDeep, cls).__new__(cls)

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        n_estimators: NonNegativeInt,
        internal_size: NonNegativeInt,
        max_depth: NonNegativeInt,
        feature_fraction: NonNegativeFloat,
    ) -> None:
        """Initialize shared estimator hyperparameters."""
        self.feature_names_: list[str] = []

        self.n_estimators = n_estimators
        self.internal_size = internal_size
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction

    @validate_call(config={"arbitrary_types_allowed": True})
    def _validate_X(self, X: Inputs) -> Inputs:
        """Validate and convert features to a numeric, finite numpy array.

        If ``X`` is a pandas DataFrame and the estimator has ``feature_names_`` set,
        the DataFrame is re-ordered/selected according to those names.

        Args:
            X: Feature matrix.

        Returns:
            A numpy array suitable for Keras.
        """
        if isinstance(X, pd.DataFrame):
            cols = getattr(self, "feature_names_", []) or list(X.columns)
            arr = check_array(
                np.array(X[cols]),
                dtype="numeric",
                ensure_all_finite=True,
            )
            return np.asarray(arr, dtype=np.float64)

        arr = check_array(
            np.array(X),
            dtype="numeric",
            ensure_all_finite=True,
        )
        return np.asarray(arr, dtype=np.float64)

    @abstractmethod
    def _validate_y(self, y: GroundTruth) -> GroundTruth:
        """Validate targets.

        Subclasses should enforce the expected shape and dtype constraints for the
        particular learning problem.
        """
        raise NotImplementedError("Subclasses must implement _validate_y method.")
