"""Unit tests for the Keras DeepTree layer."""

import pytest

tf = pytest.importorskip("tensorflow")
from tree_machine.deep_trees.layers import DeepTree


def test_mu_sums_to_one() -> None:
    """Path probabilities across leaves should sum to 1 for each sample."""
    layer = DeepTree(depth=3, feature_sample=0.8, output_size=2)
    x = tf.random.normal((5, 4))
    _ = layer(x)

    mu = layer._path_probabilities(x)
    row_sums = tf.reduce_sum(mu, axis=1)

    tf.debugging.assert_near(row_sums, tf.ones_like(row_sums), atol=1e-6)


def test_depth1_matches_manual_formula() -> None:
    """For depth=1, output should match the manual 0.5*pi0 + 0.5*pi1 formula."""
    layer = DeepTree(depth=1, feature_sample=1.0, output_size=1)

    x = tf.constant([[0.0, 0.0], [1.0, -1.0]], dtype=tf.float32)
    _ = layer(x)

    layer.pi.assign(tf.constant([[10.0], [20.0]], dtype=tf.float32))

    decision_dense = layer.decision
    decision_dense.set_weights(
        [
            tf.zeros_like(decision_dense.get_weights()[0]),
            tf.zeros_like(decision_dense.get_weights()[1]),
        ]
    )

    y = layer(x)
    expected = 0.5 * 10.0 + 0.5 * 20.0

    tf.debugging.assert_near(y, tf.fill((2, 1), expected), atol=1e-6)


def test_feature_sample_never_zero_features() -> None:
    """Very small feature_sample should still keep at least one feature."""
    layer = DeepTree(depth=2, feature_sample=1e-9, output_size=1)
    x = tf.random.normal((3, 10))
    _ = layer(x)

    mask = layer.features_mask
    assert int(mask.shape[0]) >= 1
    assert int(mask.shape[1]) == 10
