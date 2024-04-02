"""
Definitions for a deep tree regressor.
"""

import keras.losses as kl  # type: ignore
import keras.metrics as km  # type: ignore
import numpy as np
from keras.models import Model  # type: ignore
from numpy.typing import NDArray
from shap import DeepExplainer  # type: ignore
from sklearn.base import RegressorMixin  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from ..types import Actuals, Inputs, Predictions
from .base import BaseDeep
from .layers.builder import DeepTreeBuilder

_losses: dict[str, tuple] = {
    "mae": (kl.MeanAbsoluteError, km.MeanAbsoluteError),
    "mse": (kl.MeanSquaredError, km.MeanSquaredError),
    "mape": (kl.MeanAbsolutePercentageError, km.MeanAbsolutePercentageError),
}


class DeepTreeRegressor(BaseDeep, RegressorMixin):
    """
    Defines a deep tree regressor.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        internal_size: int = 12,
        max_depth: int = 6,
        feature_fraction: float = 1.0,
        loss: str = "mse",
        explain_fraction: float = 0.2,
        alpha_l1: float = 0.0,
        lambda_l2: float = 0.0,
    ) -> None:
        """
        Constructor for DeepTreeRegressor.
        See BaseDeepTree for more details.

        Args:
            loss: Specific loss function to use in the estimator. Defaults to "mse".
            Accepts "mse", "mae", "mape".
        """
        super().__init__(
            "regression",
            n_estimators,
            internal_size,
            max_depth,
            feature_fraction,
            explain_fraction,
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
        X_, y_ = self._pre_fit(X, y)
        inputs, outputs = DeepTreeBuilder(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
            self.alpha_l1,
            self.lambda_l2,
            arch_type="regression",
        )(X_.shape[1], self.internal_size, y_.shape[1])

        self.model_ = Model(inputs=inputs, outputs=outputs)
        self.model_.compile(
            loss=_losses[self.loss][0](),
            optimizer="adam",
            metrics=[
                _losses[self.loss][1](),
            ],
        )
        self.model_.fit(X_, y_, **fit_params)

        if BaseDeep._tf_version < (2, 16, 0):
            self.explainer_ = DeepExplainer(
                self.model_,
                X_[
                    np.random.randint(
                        X_.shape[0], size=int(X.shape[0] * self.explain_fraction)
                    ),
                    :,
                ],
            )

        return self

    def predict(self, X: Inputs) -> Predictions:
        """
        Returns model prediction.
        """
        check_is_fitted(self, "model_")
        reshaped_predict = self.model_.predict(
            self._treat_dataframe(X, self.feature_names),
        ).reshape(
            X.shape[0],
            -1,
        )

        if reshaped_predict.shape[1] == 1:
            return reshaped_predict[:, 0]
        return reshaped_predict

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
