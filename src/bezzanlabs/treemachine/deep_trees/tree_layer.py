"""
Implementation of a Deep Tree layer for continuous trees, using a tensorflow backend.
"""
import typing as tp

import tensorflow as tf  # type: ignore
from keras.layers import Dense, Layer, concatenate  # type: ignore
from numpy import arange, eye
from numpy.random import choice


class DeepTree(Layer):
    """
    Class to be able to train a tree-like structure using backpropagation.
    """

    features_mask: tf.Variable

    def __init__(
        self,
        depth: int = 3,
        feature_sample: float = 0.5,
        output_size: int = 2,
        **kwargs,
    ) -> None:
        """
        Constructor for DeepTree layer.

        Args:
            depth: Depth of the tree to use
            feature_sample: Sample ratio to use when fitting the tree.
            output_size: Number of output neurons to use as outputs.
            kwargs: Any arguments to pass to tf.keras.layers.Layer constructor.
        """
        super().__init__(**kwargs)

        self.depth = depth
        self.feature_sample = feature_sample
        self.output_size = output_size

        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=(2**self.depth, self.output_size)
            ),
            dtype="float32",
            trainable=True,
            name="leaves_variable",
        )

        self.decision = Dense(
            units=2**self.depth,
            activation="sigmoid",
            name="decision_variable",
        )

    def get_config(self) -> dict[str, tp.Any]:
        """
        Returns config of the layer to help with serialization.
        """
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "feature_sample": self.feature_sample,
                "output_size": self.output_size,
            }
        )
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds layer based on weights.

        input_shape = (batch, variables)
        """
        input_dim = input_shape[1]
        num_used_features = int(input_dim * self.feature_sample)
        one_hot = eye(input_dim)
        sampled_feature_indexes = choice(
            arange(input_dim), num_used_features, replace=False
        )

        self.features_mask = tf.Variable(
            one_hot[sampled_feature_indexes],
            trainable=False,
            dtype="float32",
            name=f"{self.tree_name}_features_mask",
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Returns calculation for DeepTree.

        input_shape[0] = (None, flattened_results)
        """
        batch_size = tf.shape(inputs)[0]

        features_masked = tf.matmul(inputs, self.features_mask, transpose_b=True)
        decisions = tf.expand_dims(self.decision(features_masked), axis=2)
        decisions = concatenate([decisions, 1 - decisions], axis=2)

        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])
            mu = tf.tile(mu, (1, 1, 2))
            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, (batch_size, 2**self.depth))
        outputs = tf.matmul(mu, self.pi)

        return outputs
