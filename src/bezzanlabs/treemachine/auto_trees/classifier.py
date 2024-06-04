"""
Definition of a auto classification tree.
"""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from bezzanlabs.treemachine.types import Actuals, Inputs, Predictions

from .base import BaseAuto
from .config import classification_metrics, default_hyperparams
from .splitter_proto import SplitterLike


class _Identity(TransformerMixin):
    """
    Performs an identity transformation on the data it receives.
    """

    def fit(self, X: Inputs, y: Actuals) -> "_Identity":
        return self

    def transform(self, X: Inputs) -> Inputs:
        return X


class Classifier(BaseAuto, ClassifierMixin):
    """
    Defines a auto classifier tree. Uses bayesian optimisation to select a set of
    hyperparameters automatically, and accepts user intervention over the parameters
    to be selected and their domains.
    """

    model_: XGBClassifier

    def __init__(
        self,
        metric: str = "f1",
        cv: SplitterLike = KFold(n_splits=5),
        optimisation_iter: int = 100,
    ) -> None:
        """
        Constructor for ClassificationTree.
        See BaseTree for more details.
        """

        super().__init__(
            "classification",
            metric,
            cv,
            optimisation_iter,
        )

    def fit(self, X: Inputs, y: Actuals, **fit_params) -> "Classifier":
        """
        Fits estimator using bayesian optimization to select hyperparameters.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass for the
            internal solver:
                `sampler`: specific imblearn sampler to be used in the estimation.
                `hyperparams`: dictionary containing the space to be used in the
                optimisation process.

                For all other parameters to pass to estimator, please append
                "estimator__" to their name so the pipeline can route them directly to
                the tree algorithm. If using inside another pipeline, it need to be
                appended by an extra __.
        """
        self._feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

        base_params = fit_params.pop("hyperparams", default_hyperparams)
        sampler = fit_params.pop("sampler", _Identity())
        timeout = fit_params.pop("timeout", 180)

        optimiser = self._create_optimiser(
            pipe=Pipeline(
                [
                    ("sampler", sampler),
                    ("estimator", XGBClassifier(n_jobs=-1)),
                ]
            ),
            params={f"estimator__{key}": base_params[key] for key in base_params},
            metric=make_scorer(
                classification_metrics.get(self.metric, "f1"),
            ),
            timeout=timeout,
        )

        optimiser.fit(
            self._treat_x(X),
            self._treat_y(y),
            **fit_params,
        )

        self.model_ = optimiser.best_estimator_.steps[1][1]
        self.best_params_ = optimiser.best_params_
        self.trials_ = optimiser.trials_
        self.feature_importances_ = self.model_.feature_importances_

        return self

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Returns model "probability" prediction.
        """
        check_is_fitted(self, "model_")

        return self.model_.predict_proba(self._treat_x(X))

    def score(
        self,
        X: Inputs,
        y: Actuals,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Returns model score.
        """
        return classification_metrics.get(self.metric, "f1")(
            self._treat_y(y),
            self.predict(X) if self.metric != "auc" else self.predict_proba(X),
            sample_weight=sample_weight,
        )
