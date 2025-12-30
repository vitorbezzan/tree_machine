"""Unit tests for the Keras DeepForest layer."""

import pickle

import pytest

tf = pytest.importorskip("tensorflow")
from tree_machine.deep_trees.layers import DeepForest


def _force_depth1_tree_constant_output(tree, leaf0: float, leaf1: float) -> None:
    """Force a depth-1 tree to output a deterministic constant for any input.

    The tree is configured so that the split probability is always 0.5 and both leaf
    values are fixed.
    """
    tree.pi.assign(tf.constant([[leaf0], [leaf1]], dtype=tf.float32))

    decision_dense = tree.decision
    decision_dense.set_weights(
        [
            tf.zeros_like(decision_dense.get_weights()[0]),
            tf.zeros_like(decision_dense.get_weights()[1]),
        ]
    )


def test_forest_output_shape_and_dtype() -> None:
    """DeepForest should return a float32 tensor of shape (n_samples, output_size)."""
    layer = DeepForest(n_trees=3, depth=2, feature_sample=0.7, output_size=4)
    x = tf.random.normal((5, 10), dtype=tf.float32)

    y = layer(x)

    assert tuple(y.shape) == (5, 4)
    assert y.dtype == tf.float32
    tf.debugging.assert_all_finite(y, "DeepForest output contains non-finite values")


def test_depth1_sum_of_trees_matches_manual_formula() -> None:
    """For depth=1, a forced 0.5 split should match the manual expected sum."""
    forest = DeepForest(n_trees=2, depth=1, feature_sample=1.0, output_size=1)

    x = tf.constant([[0.0, 0.0], [1.0, -1.0]], dtype=tf.float32)
    _ = forest(x)

    _force_depth1_tree_constant_output(forest.ensemble[0], leaf0=10.0, leaf1=20.0)
    _force_depth1_tree_constant_output(forest.ensemble[1], leaf0=1.0, leaf1=3.0)

    y = forest(x)
    expected = 15.0 + 2.0

    tf.debugging.assert_near(y, tf.fill((2, 1), expected), atol=1e-6)


def test_feature_sample_never_zero_features_in_each_tree() -> None:
    """Each individual tree should keep at least one feature, regardless of fraction."""
    layer = DeepForest(n_trees=4, depth=2, feature_sample=1e-9, output_size=1)
    x = tf.random.normal((3, 10))
    _ = layer(x)

    for tree in layer.ensemble:
        mask = tree.features_mask
        assert int(mask.shape[0]) >= 1
        assert int(mask.shape[1]) == 10


def test_config_roundtrip_layer_is_callable() -> None:
    """Layer config should roundtrip through get_config/from_config."""
    layer = DeepForest(n_trees=2, depth=3, feature_sample=0.5, output_size=2, name="df")
    config = layer.get_config()

    cloned = DeepForest.from_config(config)

    assert cloned.n_trees == layer.n_trees
    assert cloned.depth == layer.depth
    assert cloned.feature_sample == layer.feature_sample
    assert cloned.output_size == layer.output_size

    x = tf.random.normal((2, 7))
    y = cloned(x)
    assert tuple(y.shape) == (2, 2)


def test_pickle_roundtrip_preserves_outputs_when_weights_are_deterministic() -> None:
    """Pickling/unpickling should preserve outputs when weights are deterministic."""
    forest = DeepForest(n_trees=2, depth=1, feature_sample=1.0, output_size=1)
    x = tf.constant([[0.0, 0.0], [1.0, -1.0]], dtype=tf.float32)

    _ = forest(x)

    _force_depth1_tree_constant_output(forest.ensemble[0], leaf0=10.0, leaf1=20.0)
    _force_depth1_tree_constant_output(forest.ensemble[1], leaf0=1.0, leaf1=3.0)

    before = forest(x)

    pickled = pickle.dumps(forest)
    restored = pickle.loads(pickled)

    after = restored(x)

    tf.debugging.assert_near(before, after, atol=1e-6)
