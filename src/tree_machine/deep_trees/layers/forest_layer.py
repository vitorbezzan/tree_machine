"""Differentiable forest layer.

This module implements :class:`~tree_machine.deep_trees.layers.forest_layer.DeepForest`,
a Keras layer that aggregates multiple differentiable decision trees.

The implementation is inspired by the Keras example on neural decision forests, but
is adapted to be serializable and usable within this package's API.
"""

import tensorflow as tf
from keras.layers import Layer
from keras.saving import register_keras_serializable
from tensorflow import shape, zeros

from .tree_layer import DeepTree


@register_keras_serializable(package="tree_machine")
class DeepForest(Layer):
    """A differentiable forest built as a sum of :class:`~.DeepTree` layers."""

    def __init__(
        self,
        n_trees: int,
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
        """Initialize the forest.

        Args:
            n_trees: Number of trees in the ensemble.
            depth: Depth of each tree.
            feature_sample: Fraction of input features sampled per tree.
            output_size: Output dimension produced by each tree.
            decision_l1/decision_l2: Regularization for routing Dense weights.
            leaf_l1/leaf_l2: Regularization for leaf values.
            feature_dropout: Dropout rate applied to inputs during training.
            routing_dropout: Dropout rate applied to routing probabilities during training.
        """
        super().__init__(**kwargs)

        self.n_trees = n_trees
        self.depth = depth
        self.feature_sample = feature_sample
        self.output_size = output_size

        self.decision_l1 = float(decision_l1)
        self.decision_l2 = float(decision_l2)
        self.leaf_l1 = float(leaf_l1)
        self.leaf_l2 = float(leaf_l2)
        self.feature_dropout = float(feature_dropout)
        self.routing_dropout = float(routing_dropout)

        self.ensemble = [
            DeepTree(
                self.depth,
                self.feature_sample,
                self.output_size,
                decision_l1=self.decision_l1,
                decision_l2=self.decision_l2,
                leaf_l1=self.leaf_l1,
                leaf_l2=self.leaf_l2,
                feature_dropout=self.feature_dropout,
                routing_dropout=self.routing_dropout,
                name=f"{i}_deep_tree",
            )
            for i in range(self.n_trees)
        ]

    def get_config(self) -> dict[str, object]:
        """Return the Keras-serializable configuration for this layer."""
        config = super().get_config()
        config.update(
            {
                "n_trees": self.n_trees,
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

    def call(
        self, inputs: tf.Tensor, training: bool | None = None, *args, **kwargs
    ) -> tf.Tensor:
        """Compute the forest output for a batch.

        The result is the sum of outputs from all trees.

        Args:
            inputs: Tensor of shape ``(batch, n_features)``.

        Returns:
            Tensor of shape ``(batch, output_size)``.
        """
        batch_size = shape(inputs)[0]
        outputs = zeros([batch_size, self.output_size], dtype=inputs.dtype)

        for tree in self.ensemble:
            outputs += tree(inputs, training=training)

        return outputs

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build all trees contained in the forest."""
        for tree in self.ensemble:
            tree.build(input_shape)

        super().build(input_shape)
