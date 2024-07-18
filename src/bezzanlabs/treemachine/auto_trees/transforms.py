"""
Useful transformations needed for our classes.
"""
from sklearn.base import TransformerMixin

from .types import Actuals, Inputs


class Identity(TransformerMixin):
    """
    Performs an identity transformation on the data it receives.
    """

    def __init__(self):
        pass

    def fit(self, X: Inputs, y: Actuals) -> "Identity":
        return self

    def transform(self, X: Inputs, y=None) -> Inputs:
        return X
