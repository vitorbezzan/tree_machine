"""
Defines builder class for deep trees.
"""
import typing as tp

from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass
from tensorflow.keras.layers import Dense, Input, Layer

from .forest_layer import DeepForest


@dataclass
class DeepTreeBuilder:
    """
    Builds a deep tree architecture, with all layers, inputs and outputs. Validates
        inputs, and returns input and output layers.
    """

    n_estimators: PositiveInt
    max_depth: PositiveInt
    feature_fraction: PositiveFloat
    alpha_l1: NonNegativeFloat
    lambda_l2: NonNegativeFloat
    tree_type: tp.Literal["classification", "regression"] = "classification"

    def get_tree(
        self,
        inputs_size: int,
        internal_size: int,
        output_size: int,
    ) -> tuple[Layer, Layer]:
        """
        Builds a deep tree architecture.
        """
        inputs = tp.cast(Layer, Input(shape=(inputs_size,)))
        forest = DeepForest(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
            internal_size,
            self.alpha_l1,
            self.lambda_l2,
            name="forest_layer",
        )(inputs)

        if self.tree_type == "classification":
            return inputs, tp.cast(
                Layer, Dense(output_size, activation="softmax")(forest)
            )

        return inputs, tp.cast(Layer, Dense(output_size, activation="linear")(forest))
