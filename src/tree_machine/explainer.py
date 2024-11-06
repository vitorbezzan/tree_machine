"""
ExplainerMixIn, to specify classes that implement explanability using shap.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from shap import Explainer

from .types import Inputs


class ExplainerMixIn(ABC):
    """
    Implements minimal interface for explainability using shap values.
    """

    explainer_: Explainer

    @abstractmethod
    def explain(self, X: Inputs, **explainer_params) -> dict[str, NDArray[np.float64]]:
        """
        Returns a dictionary containing the average response values and their respective
         shap values (for each class if we are using a multi-classifier).
        """
        raise NotImplementedError()
