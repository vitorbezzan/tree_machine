"""Deep-forest regression estimator built on Keras.

This module provides :class:`~tree_machine.deep_trees.regression.DFRegression`, an
sklearn-compatible regressor that builds a differentiable forest model using Keras.

The estimator follows the familiar ``fit``/``predict``/``score`` API and supports
both built-in and user-provided Keras losses/metrics.
"""

import typing as tp

import numpy as np
import pandas as pd
import tensorflow.keras.losses as kl
import tensorflow.keras.metrics as km
from numpy.typing import NDArray
from pydantic import AfterValidator, validate_call
from pydantic.types import Annotated
from sklearn.base import RegressorMixin
from sklearn.utils.validation import _check_y, check_is_fitted
from tensorflow.keras import Model

from tree_machine.types import GroundTruth, Inputs, Predictions

from .base import BaseDeep
from .layers import DeepForestBuilder

_losses: dict[str, tuple] = {
    "mae": (kl.MeanAbsoluteError, km.MeanAbsoluteError),
    "mse": (kl.MeanSquaredError, km.MeanSquaredError),
    "mape": (kl.MeanAbsolutePercentageError, km.MeanAbsolutePercentageError),
}


def _is_acceptable_metric(loss: str) -> str:
    """Validate that a metric key is one of the supported built-ins."""
    assert loss in _losses
    return loss


AcceptableMetric = Annotated[str, AfterValidator(_is_acceptable_metric)]

LossLike = tp.Union[tp.Callable[..., tp.Any], kl.Loss]
MetricLike = tp.Union[tp.Callable[..., tp.Any], km.Metric]


def _instantiate_if_class(obj: tp.Any) -> tp.Any:
    """Instantiate ``obj`` if it is a class/type; otherwise return it unchanged."""
    if isinstance(obj, type):
        return obj()
    return obj


def _resolve_loss_and_metrics(
    *,
    metric_key: AcceptableMetric,
    loss: LossLike | None,
    metrics: tp.Sequence[MetricLike] | None,
) -> tuple[tp.Any, list[tp.Any]]:
    """Resolve the effective loss and metrics used for ``Model.compile``.

    Resolution rules:
        * If both ``loss`` and ``metrics`` are ``None``, use the built-in loss and a
          single built-in metric derived from ``metric_key``.
        * If ``loss`` is provided, use it (instantiating it when a class is passed).
          ``metrics`` defaults to an empty list unless provided.
        * If ``metrics`` is provided but ``loss`` is ``None``, use the built-in loss
          derived from ``metric_key`` and the provided metrics.

    Returns:
        A tuple ``(loss_obj, metrics_list)`` suitable for ``Model.compile``.
    """

    if loss is None and metrics is None:
        loss_obj = _losses[metric_key][0]()
        metrics_list = [_losses[metric_key][1]()]
        return loss_obj, metrics_list

    if loss is None:
        loss_obj = _losses[metric_key][0]()
    else:
        loss_obj = _instantiate_if_class(loss)

    if metrics is None:
        metrics_list = []
    else:
        metrics_list = [_instantiate_if_class(m) for m in metrics]

    return loss_obj, metrics_list


class DFRegression(BaseDeep, RegressorMixin):
    """A deep-forest regressor with an sklearn-compatible API.

    Parameters:
        metric:
            Name of the built-in metric/loss key. Must be one of ``"mae"``, ``"mse"``,
            or ``"mape"``.
        n_estimators:
            Number of trees/estimators to build.
        internal_size:
            Internal representation size used by the differentiable tree layers.
        max_depth:
            Maximum depth of each differentiable tree.
        feature_fraction:
            Fraction of features sampled per estimator.
        loss:
            Optional custom Keras loss (callable/instance/class). If omitted (and
            ``metrics`` is also omitted), the built-in loss derived from ``metric``
            is used.
        metrics:
            Optional sequence of custom Keras metrics (callables/instances/classes).
        compile_kwargs:
            Extra keyword arguments forwarded to ``Model.compile``. If
            ``compile_kwargs`` does not define an optimizer, ``"adam"`` is used.
        decision_l1/decision_l2:
            L1/L2 regularization strength applied to routing Dense weights.
        leaf_l1/leaf_l2:
            L1/L2 regularization strength applied to leaf values.
        feature_dropout:
            Dropout rate applied to inputs during training.
        routing_dropout:
            Dropout rate applied to routing probabilities during training.

    Notes:
        * Each call to :meth:`fit` builds and compiles a new Keras model.
        * :meth:`score` returns the negative loss value on ``(X, y)``.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: AcceptableMetric,
        n_estimators: int,
        internal_size: int,
        max_depth: int,
        feature_fraction: float,
        *,
        loss: LossLike | None = None,
        metrics: tp.Sequence[MetricLike] | None = None,
        compile_kwargs: dict[str, tp.Any] | None = None,
        decision_l1: float = 0.0,
        decision_l2: float = 0.0,
        leaf_l1: float = 0.0,
        leaf_l2: float = 0.0,
        feature_dropout: float = 0.0,
        routing_dropout: float = 0.0,
    ) -> None:
        """Create a :class:`DFRegression` instance."""
        super().__init__(
            n_estimators,
            internal_size,
            max_depth,
            feature_fraction,
        )

        self.metric = metric
        self.loss = loss
        self.metrics = list(metrics) if metrics is not None else None
        self.compile_kwargs = (
            dict(compile_kwargs) if compile_kwargs is not None else None
        )

        self.decision_l1 = float(decision_l1)
        self.decision_l2 = float(decision_l2)
        self.leaf_l1 = float(leaf_l1)
        self.leaf_l2 = float(leaf_l2)
        self.feature_dropout = float(feature_dropout)
        self.routing_dropout = float(routing_dropout)

        self._resolved_loss_: tp.Any | None = None
        self._resolved_metrics_: list[tp.Any] | None = None

    def fit(self, X: Inputs, y: GroundTruth, **fit_params) -> "DFRegression":
        """Fit the regressor.

        The model is built through :class:`~tree_machine.deep_trees.layers.DeepForestBuilder`
        and compiled with the resolved loss/metrics.

        Args:
            X: Feature matrix.
            y: Numeric regression targets.
            fit_params: Extra keyword arguments forwarded to ``Model.fit``.

        Returns:
            ``self``.
        """
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else []

        y_ = self._validate_y(y).reshape(X.shape[0], -1)

        inputs, outputs = DeepForestBuilder(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
            "regression",
            decision_l1=self.decision_l1,
            decision_l2=self.decision_l2,
            leaf_l1=self.leaf_l1,
            leaf_l2=self.leaf_l2,
            feature_dropout=self.feature_dropout,
            routing_dropout=self.routing_dropout,
        ).get_tree(
            X.shape[1],
            self.internal_size,
            y_.shape[1],
        )

        self.model_ = Model(inputs=inputs, outputs=outputs)

        loss_obj, metrics_list = _resolve_loss_and_metrics(
            metric_key=self.metric,
            loss=self.loss,
            metrics=self.metrics,
        )
        self._resolved_loss_ = loss_obj
        self._resolved_metrics_ = metrics_list

        compile_kwargs = (
            dict(self.compile_kwargs) if self.compile_kwargs is not None else {}
        )
        optimizer = compile_kwargs.pop("optimizer", "adam")
        self.model_.compile(
            loss=loss_obj,
            optimizer=optimizer,
            metrics=metrics_list,
            **compile_kwargs,
        )
        self.model_.fit(self._validate_X(X), y_, **fit_params)
        return self

    def predict(self, X: Inputs) -> Predictions:
        """Predict regression targets.

        Args:
            X: Feature matrix.

        Returns:
            A 1D array of predictions with shape ``(n_samples,)``.
        """
        check_is_fitted(self, "model_")

        return self.model_.predict(
            self._validate_X(X),
        ).reshape(-1)

    def score(
        self,
        X: Inputs,
        y: GroundTruth,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """Return the negative loss on ``(X, y)``.

        This matches sklearn's convention that higher scores are better.

        Args:
            X: Feature matrix.
            y: Ground-truth regression targets.
            sample_weight: Optional per-sample weights.

        Returns:
            The negative loss value as a Python ``float``.
        """
        y_true = self._validate_y(y).reshape(np.array(X).shape[0], -1)
        y_pred = np.array(self.predict(X)).reshape(y_true.shape)

        if getattr(self, "_resolved_loss_", None) is None:
            loss_obj, _ = _resolve_loss_and_metrics(
                metric_key=self.metric,
                loss=self.loss,
                metrics=self.metrics,
            )
        else:
            loss_obj = tp.cast(tp.Any, self._resolved_loss_)

        if sample_weight is None:
            loss_value = loss_obj(y_true, y_pred)
        else:
            try:
                loss_value = loss_obj(y_true, y_pred, sample_weight=sample_weight)
            except TypeError:
                loss_value = loss_obj(y_true, y_pred)

        return -loss_value.numpy()

    def _validate_y(self, y: GroundTruth) -> GroundTruth:
        """Validate regression targets.

        Targets must be numeric and one-dimensional; multi-output targets are rejected.
        """
        return _check_y(np.array(y), multi_output=False, y_numeric=True)


__all__ = [
    "DFRegression",
]
