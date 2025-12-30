"""Differentiable ("deep") decision-tree layer.

This module defines :class:`~tree_machine.deep_trees.layers.tree_layer.DeepTree`, a
Keras layer implementing a soft binary decision tree of fixed depth.

Internal nodes produce routing probabilities via a sigmoid, leaf values are
trainable, and the final output is the expected leaf value under the path
probabilities.
"""

from __future__ import annotations

import tensorflow as tf
from keras.layers import Dense, Layer, concatenate
from keras.regularizers import L1L2, Regularizer
from keras.saving import register_keras_serializable
from numpy import arange, eye
from numpy.random import choice


@register_keras_serializable(package="tree_machine")
class DeepTree(Layer):
    """A differentiable decision tree implemented as a Keras layer.

    Regularization and stochasticity knobs:
        * ``decision_l1``/``decision_l2`` apply weight decay to the routing Dense.
        * ``leaf_l1``/``leaf_l2`` apply weight decay to the leaf values ``pi``.
        * ``feature_dropout`` drops input features (like classic feature dropout).
        * ``routing_dropout`` drops/attenuates routing probabilities, encouraging
          less brittle splits.

    Notes:
        Dropout is applied only when ``training=True``.
    """

    features_mask: tf.Variable

    def __init__(
        self,
        depth: int,
        feature_sample: float,
        output_size: int,
        *,
        decision_l1: float = 0.0,
        decision_l2: float = 0.0,
        leaf_l1: float = 0.0,
        leaf_l2: float = 0.0,
        feature_dropout: float = 0.0,
        routing_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize the tree.

        Args:
            depth: Number of routing levels. Must be >= 1.
            feature_sample: Fraction of features used by the tree. Must be in (0, 1].
            output_size: Output dimension of each leaf. Must be >= 1.
            decision_l1: L1 regularization applied to routing Dense weights.
            decision_l2: L2 regularization applied to routing Dense weights.
            leaf_l1: L1 regularization applied to leaf values.
            leaf_l2: L2 regularization applied to leaf values.
            feature_dropout: Dropout rate applied to inputs during training.
            routing_dropout: Dropout rate applied to routing probabilities during training.
        """
        super().__init__(**kwargs)

        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not (0.0 < feature_sample <= 1.0):
            raise ValueError("feature_sample must be in (0, 1]")
        if output_size < 1:
            raise ValueError("output_size must be >= 1")

        for name, rate in (
            ("feature_dropout", feature_dropout),
            ("routing_dropout", routing_dropout),
        ):
            if not (0.0 <= float(rate) < 1.0):
                raise ValueError(f"{name} must be in [0, 1)")

        self.depth = int(depth)
        self.feature_sample = float(feature_sample)
        self.output_size = int(output_size)

        self.decision_l1 = float(decision_l1)
        self.decision_l2 = float(decision_l2)
        self.leaf_l1 = float(leaf_l1)
        self.leaf_l2 = float(leaf_l2)
        self.feature_dropout = float(feature_dropout)
        self.routing_dropout = float(routing_dropout)

        self.num_leaves = 2**self.depth
        self.num_internal_nodes = self.num_leaves - 1
        self.pi: tf.Variable | None = None

        decision_reg: Regularizer | None = None
        if self.decision_l1 > 0.0 or self.decision_l2 > 0.0:
            decision_reg = L1L2(l1=self.decision_l1, l2=self.decision_l2)

        self.decision = Dense(
            units=self.num_internal_nodes,
            activation="sigmoid",
            kernel_regularizer=decision_reg,
            bias_regularizer=decision_reg,
            name="decision_variable",
        )

    def get_config(self) -> dict[str, object]:
        """Return the Keras-serializable configuration for this layer."""
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "feature_sample": self.feature_sample,
                "output_size": self.output_size,
                "decision_l1": self.decision_l1,
                "decision_l2": self.decision_l2,
                "leaf_l1": self.leaf_l1,
                "leaf_l2": self.leaf_l2,
                "feature_dropout": self.feature_dropout,
                "routing_dropout": self.routing_dropout,
            }
        )
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the per-tree feature mask and leaf parameters.

        Args:
            input_shape: TensorShape whose last dimension is the number of features.
        """
        input_dim = int(input_shape[-1])
        if input_dim <= 0:
            raise ValueError("Input dimension must be a positive integer")

        num_used_features = max(1, min(input_dim, int(input_dim * self.feature_sample)))

        one_hot = eye(input_dim)
        sampled_feature_indexes = choice(
            arange(input_dim), num_used_features, replace=False
        )

        self.features_mask = tf.Variable(
            one_hot[sampled_feature_indexes],
            trainable=False,
            dtype="float32",
            name="features_mask",
        )

        leaf_reg: Regularizer | None = None
        if self.leaf_l1 > 0.0 or self.leaf_l2 > 0.0:
            leaf_reg = L1L2(l1=self.leaf_l1, l2=self.leaf_l2)

        if self.pi is None:
            self.pi = self.add_weight(
                name="leaves_variable",
                shape=(self.num_leaves, self.output_size),
                initializer=tf.random_normal_initializer(),
                regularizer=leaf_reg,
                trainable=True,
                dtype="float32",
            )

        self.decision.build(tf.TensorShape((input_shape[0], num_used_features)))
        super().build(input_shape)

    def _maybe_feature_dropout(
        self, inputs: tf.Tensor, *, training: bool | None
    ) -> tf.Tensor:
        if not training or self.feature_dropout <= 0.0:
            return inputs
        return tf.nn.dropout(inputs, rate=self.feature_dropout)

    def _maybe_routing_dropout(
        self, probs: tf.Tensor, *, training: bool | None
    ) -> tf.Tensor:
        """Apply dropout to routing probabilities and renormalize.

        ``probs`` is expected to have shape (batch, n_internal, 2).
        """
        if not training or self.routing_dropout <= 0.0:
            return probs

        dropped = tf.nn.dropout(probs, rate=self.routing_dropout)
        denom = tf.reduce_sum(dropped, axis=2, keepdims=True)
        # If both branches were dropped for a node, fall back to uniform routing.
        uniform = tf.fill(tf.shape(dropped), tf.cast(0.5, dropped.dtype))
        safe = tf.where(denom > 0.0, dropped / denom, uniform)
        return safe

    def _path_probabilities(
        self, inputs: tf.Tensor, *, training: bool | None = None
    ) -> tf.Tensor:
        """Compute path probabilities to each leaf.

        Args:
            inputs: Tensor of shape ``(batch, n_features)``.

        Returns:
            A tensor ``mu`` of shape ``(batch, num_leaves)`` where each row sums to 1.
        """
        batch_size = tf.shape(inputs)[0]

        inputs = self._maybe_feature_dropout(inputs, training=training)

        features_masked = tf.matmul(inputs, self.features_mask, transpose_b=True)

        decisions = tf.expand_dims(self.decision(features_masked), axis=2)
        decisions = concatenate([decisions, 1.0 - decisions], axis=2)
        decisions = self._maybe_routing_dropout(decisions, training=training)

        mu = tf.ones([batch_size, 1, 1], dtype=decisions.dtype)

        begin_idx = 0
        end_idx = 1
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])
            mu = tf.tile(mu, (1, 1, 2))

            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, (batch_size, self.num_leaves))
        return mu

    def call(
        self, inputs: tf.Tensor, training: bool | None = None, *args, **kwargs
    ) -> tf.Tensor:
        """Compute the expected leaf output for the given batch."""
        if self.pi is None:
            raise RuntimeError("DeepTree layer was called before being built")

        return tf.matmul(self._path_probabilities(inputs, training=training), self.pi)
