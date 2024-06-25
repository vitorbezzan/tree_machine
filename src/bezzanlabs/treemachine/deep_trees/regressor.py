"""
Definitions for a deep tree regressor.
"""
import numpy as np
import pandas as pd
import tensorflow.keras.losses as kl
import tensorflow.keras.metrics as km
from numpy.typing import NDArray
from pydantic import AfterValidator, validate_call
from pydantic.types import Annotated
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from tensorflow.keras import Model

from ..types import Actuals, Inputs, Predictions
from .base import BaseDeep
from .layers.builder import DeepTreeBuilder

_losses: dict[str, tuple] = {
    "mae": (kl.MeanAbsoluteError, km.MeanAbsoluteError),
    "mse": (kl.MeanSquaredError, km.MeanSquaredError),
    "mape": (kl.MeanAbsolutePercentageError, km.MeanAbsolutePercentageError),
}


def _is_acceptable_loss(loss: str) -> str:
    assert loss in _losses
    return loss


AcceptableLoss = Annotated[str, AfterValidator(_is_acceptable_loss)]


class DeepTreeRegressor(BaseDeep, RegressorMixin):
    """
    Defines a deep tree regressor.
    """

    @validate_call
    def __init__(
        self,
        n_estimators: int = 100,
        internal_size: int = 12,
        max_depth: int = 6,
        feature_fraction: float = 1.0,
        alpha_l1: float = 0.0,
        lambda_l2: float = 0.0,
        loss: AcceptableLoss = "mse",
    ) -> None:
        """
        Constructor for DeepTreeRegressor.
        """
        super().__init__(
            n_estimators,
            internal_size,
            max_depth,
            feature_fraction,
            alpha_l1,
            lambda_l2,
        )

        # Elements will be the respective functions/classes
        self.loss = loss

    def fit(self, X: Inputs, y: Actuals, **fit_params) -> "DeepTreeRegressor":
        """
        Fits estimator using bayesian optimization to select hyperparameters.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass for the model
                `fit` method.
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

        X_, y_ = np.array(X), np.array(y).reshape(X.shape[0], -1)

        builder = DeepTreeBuilder(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
            self.alpha_l1,
            self.lambda_l2,
            "regression",
        )
        inputs, outputs = builder.get_tree(
            X_.shape[1],
            self.internal_size,
            y_.shape[1],
        )

        self.model_ = Model(inputs=inputs, outputs=outputs)
        self.model_.compile(
            loss=_losses[self.loss][0](),
            optimizer="adam",
            metrics=[
                _losses[self.loss][1](),
            ],
        )
        self.model_.fit(X_, y_, **fit_params)
        return self

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model prediction.
        """
        check_is_fitted(self, "model_")
        return self.model_.predict(
            self._treat_x(X),
        ).reshape(-1)

    def score(
        self,
        X: Inputs,
        y: Actuals,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Returns model score.
        """
        return -_losses[self.loss][0]()(
            np.array(y).reshape(X.shape[0], -1),
            self.predict(X),
            sample_weight=sample_weight,
        ).numpy()


__all__ = [
    "DeepTreeRegressor",
]
