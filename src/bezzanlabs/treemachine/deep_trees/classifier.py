"""
Definitions for a deep tree classifier.
"""
import numpy as np
from keras.losses import CategoricalCrossentropy  # type: ignore
from keras.models import Model  # type: ignore
from numpy.typing import NDArray
from shap import DeepExplainer  # type: ignore
from sklearn.base import ClassifierMixin  # type: ignore
from sklearn.utils.validation import _check_y, check_is_fitted  # type: ignore

from ..types import Actuals, Inputs, Predictions
from .base import BaseDeep
from .layers.builder import DeepTreeBuilder


class DeepTreeClassifier(BaseDeep, ClassifierMixin):
    """
    Defines a deep tree classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        internal_size: int = 12,
        max_depth: int = 6,
        feature_fraction: float = 1.0,
        explain_fraction: float = 0.2,
    ) -> None:
        """
        Constructor for DeepTreeClassifier.
        See BaseDeepTree for more details.
        """
        super().__init__(
            "classification",
            n_estimators,
            internal_size,
            max_depth,
            feature_fraction,
            explain_fraction,
        )

    def fit(self, X: Inputs, y: Actuals, **fit_params) -> "DeepTreeClassifier":
        """
        Fits estimator using bayesian optimization to select hyperparameters.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass for the model
            `fit` method.
        """
        X_, y_ = self._pre_fit(X, self._treat_y(y))
        inputs, outputs = DeepTreeBuilder(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
        )(X_.shape[1], self.internal_size, self.labeler.classes_.shape[0])

        self.model_ = Model(inputs=inputs, outputs=outputs)
        self.model_.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[
                "categorical_crossentropy",
            ],
        )
        self.model_.fit(X_, y_, **fit_params)
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
        Returns model prediction. For regression returns the regression values, and for
        classification, there is an override that returns the class predictions.
        """
        check_is_fitted(self, "model_")
        predictions = self.model_.predict(
            self._treat_dataframe(X, self.feature_names),
        )

        return np.array(
            self.labeler.inverse_transform(
                np.where(predictions == predictions.max(axis=1).reshape(-1, 1), 1, 0)
            ),
        ).reshape(-1)

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Returns probability for each class.
        Only available for classification tasks.
        """
        check_is_fitted(self, "model_")

        return self.model_.predict(
            self._treat_dataframe(X, self.feature_names),
        ).reshape(X.shape[0], -1)

    def score(
        self,
        X: Inputs,
        y: Actuals,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Returns model score.
        """
        return -CategoricalCrossentropy()(
            self.labeler.transform(np.array(self._treat_y(y)).reshape(-1, 1)),
            self.predict_proba(X),
            sample_weight=sample_weight,
        ).numpy()

    @staticmethod
    def _treat_y(
        y: Actuals,
    ) -> NDArray[np.float64]:
        return _check_y(y, multi_output=False)
