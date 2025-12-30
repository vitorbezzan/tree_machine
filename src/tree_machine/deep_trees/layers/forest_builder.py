"""Model-building utilities for differentiable forests.

This module defines :class:`~tree_machine.deep_trees.layers.forest_builder.DeepForestBuilder`,
a small helper that constructs an input tensor and a deep-forest backbone, then adds
an output head suitable for classification or regression.
"""

import typing as tp

from pydantic import PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass
from tensorflow.keras.layers import Dense, Input, Layer

from .forest_layer import DeepForest


@dataclass
class DeepForestBuilder:
    """Build a Keras computation graph for a differentiable forest.

    The returned tuple ``(inputs, outputs)`` can be passed directly to
    ``tensorflow.keras.Model(inputs=..., outputs=...)``.
    """

    n_estimators: PositiveInt
    max_depth: PositiveInt
    feature_fraction: PositiveFloat
    tree_type: tp.Literal["classification", "regression"] = "classification"

    # Regularization / dropout (optional)
    decision_l1: float = 0.0
    decision_l2: float = 0.0
    leaf_l1: float = 0.0
    leaf_l2: float = 0.0
    feature_dropout: float = 0.0
    routing_dropout: float = 0.0

    def get_tree(
        self,
        inputs_size: int,
        internal_size: int,
        output_size: int,
    ) -> tuple[Layer, Layer]:
        """Create input and output layers for a differentiable forest model.

        Args:
            inputs_size: Number of input features.
            internal_size: Intermediate/output size produced by the :class:`DeepForest`.
            output_size: Number of outputs for the final Dense head.

        Returns:
            A tuple ``(inputs, outputs)``.
        """
        inputs = tp.cast(Layer, Input(shape=(inputs_size,)))
        forest = DeepForest(
            self.n_estimators,
            self.max_depth,
            self.feature_fraction,
            internal_size,
            decision_l1=self.decision_l1,
            decision_l2=self.decision_l2,
            leaf_l1=self.leaf_l1,
            leaf_l2=self.leaf_l2,
            feature_dropout=self.feature_dropout,
            routing_dropout=self.routing_dropout,
            name="forest_layer",
        )(inputs)

        if self.tree_type == "classification":
            return inputs, tp.cast(
                Layer, Dense(output_size, activation="softmax")(forest)
            )

        return inputs, tp.cast(Layer, Dense(output_size, activation="linear")(forest))
