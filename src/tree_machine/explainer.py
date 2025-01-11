"""
ExplainerMixIn, to specify classes that implement explainability using shap.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .types import Inputs

try:
    from shap import Explainer

    class _ExplainerMixIn(ABC):
        """
        Implements minimal interface for explainability using shap values.
        """

        explainer_: Explainer

        @abstractmethod
        def explain(
            self, X: Inputs, **explainer_params
        ) -> dict[str, NDArray[np.float64]]:
            """
            Returns a dictionary containing the average response values and their respective
             shap values (for each class if we are using a multi-classifier).
            """
            raise NotImplementedError()

except ModuleNotFoundError:

    class _ExplainerMixIn(ABC):  # type: ignore
        """
        Implements minimal interface for explainability using shap values.
        """

        @abstractmethod
        def explain(
            self, X: Inputs, **explainer_params
        ) -> dict[str, NDArray[np.float64]]:
            """
            Returns a dictionary containing the average response values and their respective
             shap values (for each class if we are using a multi-classifier).
            """
            raise NotImplementedError()


ExplainerMixIn = _ExplainerMixIn
