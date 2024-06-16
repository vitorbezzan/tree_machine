"""
Classifier auto tree.
"""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from ..optimize import OptimizerConfig
from ..splitter_proto import SplitterLike
from ..transforms import Identity
from ..types import Actuals, Inputs, Predictions
from .base import BaseAuto
from .config import classification_metrics, default_hyperparams


class Classifier(BaseAuto, ClassifierMixin):
    """
    Defines an auto classifier tree. Uses bayesian optimisation to select a set of
    hyperparameters automatically, and accepts user intervention over the parameters
    to be selected and their domains.
    """

    model_: XGBClassifier
    feature_importances_: NDArray[np.float64]

    def __init__(
        self,
        metric: str = "f1",
        cv: SplitterLike = KFold(n_splits=5),
        optimisation_iter: int = 100,
    ) -> None:
        """
        Constructor for ClassificationTree.
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

                sampler: specific imblearn sampler to be used in the estimation.
                hyperparams: dictionary containing the space to be used in the
                    optimization process.
                timeout: timeout in seconds to use for the optimizer.

                For all other parameters to pass directly to estimator, please append
                    "estimator__" to their name so the pipeline can route them directly
                    to the tree algorithm. If using inside another pipeline, it needs
                    to be appended by extra __.
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

        base_params = fit_params.pop("hyperparams", default_hyperparams)
        timeout = fit_params.pop("timeout", 180)
        sampler = fit_params.pop("sampler", Identity())

        self._fit(
            Pipeline(
                [
                    ("sampler", sampler),
                    ("estimator", XGBClassifier(n_jobs=-1)),
                ]
            ),
            X,
            y,
            {f"estimator__{key}": base_params[key] for key in base_params},
            make_scorer(classification_metrics[self.metric], greater_is_better=True),
            OptimizerConfig(
                n_trials=self.optimisation_iter,
                timeout=timeout,
                cv=self.cv,
                return_train_score=True,
            ),
            **fit_params,
        )

        self.model_ = self.optimizer_.best_estimator_.steps[1][1]
        self.feature_importances_ = self.model_.feature_importances_

        return self

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Returns model probabilities.
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
        return classification_metrics[self.metric](
            self._treat_y(y),
            self.predict(X) if self.metric != "auc" else self.predict_proba(X),
            sample_weight=sample_weight,
        )
