"""
Defines base class for all deep trees.
"""
from dataclasses import dataclass

from keras.layers import Dense, Input, Layer  # type: ignore

from .forest_layer import DeepForest


@dataclass
class DeepTreeBuilder:
    """
    Builds a deep tree architecture.
    """

    n_estimators: int
    max_depth: int
    feature_fraction: float
    arch_type: str = "classification"

    def __call__(
        self,
        inputs_size: int,
        internal_size: int,
        output_size: int,
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
            name="forest_layer",
        )(inputs)

        if self.arch_type == "classification":
            output = Dense(output_size, activation="softmax")(forest)
        else:
            output = Dense(output_size, activation="linear")(forest)

        return inputs, output
