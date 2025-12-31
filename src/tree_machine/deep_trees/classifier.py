"""Deep-forest classification estimator built on Keras.

This module provides :class:`~tree_machine.deep_trees.classifier.DFClassifier`, an
sklearn-compatible classifier that builds a differentiable forest model using Keras.

The estimator follows the familiar ``fit``/``predict``/``predict_proba``/``score`` API
and supports both built-in and user-provided Keras losses/metrics.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import pandas as pd
import tensorflow.keras.losses as kl
import tensorflow.keras.metrics as km
from numpy.typing import NDArray
from pydantic import AfterValidator, validate_call
from pydantic.types import Annotated
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_y, check_is_fitted
from tensorflow.keras import Model

from tree_machine.types import GroundTruth, Inputs, Predictions

from .base import BaseDeep
from .layers import DeepForestBuilder

# Built-in loss/metric pairings.
_losses: dict[str, tuple] = {
    "cross_entropy": (
        kl.CategoricalCrossentropy,
        km.CategoricalCrossentropy,
    ),
}


def _is_acceptable_metric(metric: str) -> str:
    """Validate that a metric key is one of the supported built-ins."""
    assert metric in _losses
    return metric


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


class DFClassifier(BaseDeep, ClassifierMixin):
    """A deep-forest classifier with an sklearn-compatible API.

    Parameters:
        metric:
            Name of the built-in metric/loss key. Currently only
            ``"cross_entropy"`` is supported.
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

    label_encoder_: LabelEncoder
    classes_: NDArray[np.object_] | NDArray[np.int_] | NDArray[np.str_]

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        metric: AcceptableMetric = "cross_entropy",
        n_estimators: int = 100,
        internal_size: int = 12,
        max_depth: int = 6,
        feature_fraction: float = 1.0,
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

    def fit(self, X: Inputs, y: GroundTruth, **fit_params) -> "DFClassifier":
        """Fit the classifier.

        Args:
            X: Feature matrix.
            y: Class labels (binary or multiclass).
            fit_params: Extra keyword arguments forwarded to ``Model.fit``.

        Returns:
            ``self``.
        """
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else []

        y_checked = self._validate_y(y)

        # Encode classes to integer indices, then to one-hot.
        self.label_encoder_ = LabelEncoder()
        y_index = self.label_encoder_.fit_transform(np.array(y_checked).reshape(-1))
        self.classes_ = tp.cast(NDArray, self.label_encoder_.classes_)

        n_classes = int(len(self.classes_))
        y_onehot = np.eye(n_classes, dtype=np.float64)[y_index]

        inputs, outputs = DeepForestBuilder(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
            "classification",
            decision_l1=self.decision_l1,
            decision_l2=self.decision_l2,
            leaf_l1=self.leaf_l1,
            leaf_l2=self.leaf_l2,
            feature_dropout=self.feature_dropout,
            routing_dropout=self.routing_dropout,
        ).get_tree(
            np.array(X).shape[1],
            self.internal_size,
            n_classes,
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
        self.model_.fit(self._validate_X(X), y_onehot, **fit_params)
        return self

    def predict_proba(self, X: Inputs) -> Predictions:
        """Predict class probabilities.

        Returns:
            Array of shape ``(n_samples, n_classes)``.
        """
        check_is_fitted(self, "model_")
        proba = self.model_.predict(self._validate_X(X))
        return np.asarray(proba, dtype=np.float64).reshape(np.array(X).shape[0], -1)

    def predict(self, X: Inputs) -> Predictions:
        """Predict classes."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return tp.cast(Predictions, self.label_encoder_.inverse_transform(idx))

    def score(
        self,
        X: Inputs,
        y: GroundTruth,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> float:
        """Return the negative loss on ``(X, y)`` (higher is better)."""
        y_checked = self._validate_y(y)
        y_idx = self.label_encoder_.transform(np.array(y_checked).reshape(-1))
        y_true = np.eye(len(self.classes_), dtype=np.float64)[y_idx]
        y_pred = np.asarray(self.predict_proba(X), dtype=np.float64)

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
        """Validate classification targets.

        Supports binary and multiclass 1D targets; rejects multilabel indicators.
        """
        y_arr = _check_y(np.array(y), multi_output=False, y_numeric=False)
        t = type_of_target(y_arr)
        if t not in {"binary", "multiclass"}:
            raise ValueError(
                "Only binary and multiclass 1D targets are supported for DFClassifier. "
                f"Got target type: {t}."
            )
        return y_arr


__all__ = [
    "DFClassifier",
]
