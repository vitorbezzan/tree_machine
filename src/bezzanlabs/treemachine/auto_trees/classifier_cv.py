"""
Defines class to instantiate classifier trees.
"""
import typing as tp

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline
from numpy.typing import NDArray
from pydantic import NonNegativeInt, validate_call
from sklearn.base import ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from .base import BaseAutoTree
from .defaults import TUsrDistribution, defaults, get_param_distributions
from .metrics import AcceptableClassifier, classification_metrics
from .transforms import Identity
from .types import Actuals, Inputs, Predictions


class ClassifierCVOptions(tp.TypedDict, total=False):
    """
    Available options to use when fitting a classifier model.

    sampler: BaseSampler object from `imblearn` package.
    monotone_constraints: dictionary containing monotonicity direction allowed for each
        variable. 0 means no monotonicity, 1 means increasing and -1 means decreasing
        monotonicity.
    interactions: list of lists containing permitted relationships in data.
    distributions: dictionary with distribution bounds for each hyperparameter to search
        on during optimization.
    """

    sampler: BaseSampler
    monotone_constraints: dict[str, int] | dict[int, int]
    interactions: list[list[int] | list[str]]
    distributions: TUsrDistribution


class ClassifierCV(BaseAutoTree, ClassifierMixin):
    """
    Defines an auto classifier tree.
    """

    model_: XGBClassifier
    feature_importances_: NDArray[np.float64]

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: AcceptableClassifier = "f1",
        cv: BaseCrossValidator = KFold(n_splits=5),
        n_trials: NonNegativeInt = 100,
        timeout: NonNegativeInt = 180,
        n_jobs: int = -1,
    ) -> None:
        """
        Constructor for ClassifierCV.

        Args:
            metric: Metric to use as base for estimation process.
            cv: Splitter object to use when estimating the model.
            n_trials: Number of optimization trials to use when finding a model.
            timeout: Timeout in seconds to stop the optimization.
            n_jobs: Number of processes to use internally when estimating the model.
        """
        super().__init__(metric, cv, n_trials, timeout, n_jobs)

    def fit(
        self,
        X: Inputs,
        y: Actuals,
        **fit_params: ClassifierCVOptions,
    ) -> "ClassifierCV":
        """
        Fits estimator using bayesian optimization.

        Args:
            X: input data to use in fitting trees.
            y: actual targets for fitting.
            fit_params: dictionary containing specific parameters to pass to the base
                classifier or the parameter distribution.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        pipeline = [
            ("sampler", fit_params.get("sampler", Identity())),
            (
                "estimator",
                XGBClassifier(
                    n_jobs=self._n_jobs,
                    monotone_constraints=fit_params.get("monotone_constraints", None),
                    interaction_constraints=fit_params.get("interactions", None),
                ),
            ),
        ]

        distributions = get_param_distributions(
            tp.cast(TUsrDistribution, fit_params.get("distributions", defaults)),
        )

        self._fit(
            Pipeline(pipeline),
            self._treat_x(X),
            self._treat_y(y),
            {f"estimator__{k}": v for k, v in distributions.items()},
            make_scorer(classification_metrics[self._metric], greater_is_better=True),
            self._cv,
        )

        self.model_ = self.optimizer_.best_estimator_.steps[-1][1]
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
        return classification_metrics[self._metric](
            self._treat_y(y),
            self.predict(X),
            sample_weight=sample_weight,
        )
