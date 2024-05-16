"""
Defines base class for all deep trees.
"""
from dataclasses import dataclass

from keras.layers import Dense, Input, Layer

from .forest_layer import DeepForest


@dataclass
class DeepTreeBuilder:
    """
    Builds a deep tree architecture.
    """

    n_estimators: int
    max_depth: int
    feature_fraction: float
    alpha_l1: float
    lambda_l2: float
    arch_type: str = "classification"

    def __call__(
        self,
        inputs_size: int,
        internal_size: int,
        output_size: int,
        alpha_l1: float = 0.0,
        lambda_l2: float = 0.0,
    ) -> tuple[Layer, Layer]:
        """
        Builds a deep tree architecture.
        """
        inputs = Input(shape=(inputs_size,))
        forest = DeepForest(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
            internal_size,
            self.alpha_l1,
            self.lambda_l2,
            name="forest_layer",
        )(inputs)

        if self.arch_type == "classification":
            output = Dense(output_size, activation="softmax")(forest)
        else:
            output = Dense(output_size, activation="linear")(forest)

        return inputs, output
