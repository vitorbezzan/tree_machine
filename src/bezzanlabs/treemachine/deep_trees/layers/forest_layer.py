"""
Deep Forest layer - as proposed by
https://keras.io/examples/structured_data/deep_neural_decision_forests/ and changed
accordingly to make it picklable and usable compatible with package framework.
"""
import tensorflow as tf  # type: ignore
from keras.layers import Layer  # type: ignore
from keras.regularizers import L1L2  # type: ignore
from tensorflow import shape, zeros  # type: ignore

from .tree_layer import DeepTree


class DeepForest(Layer):
    """
    Class to be able to train a forest-like structure using backpropagation.
    """

    def __init__(
        self,
        n_trees: int,
        depth: int,
        feature_sample: float,
        output_size: int,
        alpha_l1: float = 0.0,
        lambda_l2: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Constructor for DeepForest layer class.

        Args:
            n_trees: Number of trees to use in the forest estimator.
            depth: Depth of trees to use in the forest estimator.
            feature_sample: Sample ratio of variables to use in each tree.
            output_size: Number of output neurons to use as output.
            alpha_l1: L1 regularization parameter. Default is 0.0.
            lambda_l2: L2 regularization parameter. Default is 0.0.
        """
        super().__init__(**kwargs)

        self.n_trees = n_trees
        self.depth = depth
        self.feature_sample = feature_sample
        self.output_size = output_size
        self.alpha_l1 = alpha_l1
        self.lambda_l2 = lambda_l2

        self.ensemble = [
            DeepTree(
                self.depth,
                self.feature_sample,
                self.output_size,
                name=f"{i}_deep_tree",
                activity_regularizer=L1L2(
                    l1=alpha_l1,
                    l2=lambda_l2,
                ),
            )
            for i in range(self.n_trees)
        ]

    def get_config(self) -> dict[str, object]:  # pragma: no cover
        """
        Returns config of the layer to help with serialization.
        """
        config = super().get_config()
        config.update(
            {
                "n_trees": self.n_trees,
                "depth": self.depth,
                "feature_sample": self.feature_sample,
                "output_size": self.output_size,
            }
        )
        return config

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Returns calculation for DeepTree.

        input_shape[0] = (None, flattened_results)
        """
        batch_size = shape(inputs)[0]
        outputs = zeros([batch_size, self.output_size])

        for tree in self.ensemble:
            outputs += tree(inputs)

        return outputs
