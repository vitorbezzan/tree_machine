"""Unit tests for the Keras DeepTree layer."""

import pytest

tf = pytest.importorskip("tensorflow")
from tree_machine.deep_trees.layers import DeepTree  # noqa


@pytest.fixture(scope="session")
def tree_layer():
    """Standard DeepTree layer."""
    return DeepTree(depth=3, feature_sample=0.8, output_size=2)


@pytest.fixture(scope="session")
def depth1_layer():
    """Depth-1 DeepTree layer for formula tests."""
    return DeepTree(depth=1, feature_sample=1.0, output_size=1)


@pytest.fixture(scope="session")
def small_feature_sample_layer():
    """DeepTree with very small feature sample."""
    return DeepTree(depth=2, feature_sample=1e-9, output_size=1)


@pytest.fixture(scope="session")
def sample_input():
    """Sample input tensor."""
    return tf.random.normal((5, 4), seed=42)


@pytest.fixture(scope="session")
def depth1_input():
    """Input for depth-1 tests."""
    return tf.constant([[0.0, 0.0], [1.0, -1.0]], dtype=tf.float32)


@pytest.fixture(scope="session")
def feature_sample_input():
    """Input for feature sample tests."""
    return tf.random.normal((3, 10), seed=42)


class TestPathProbabilities:
    """Tests for path probability computation."""

    def test_mu_sums_to_one(self, tree_layer, sample_input):
        """Path probabilities across leaves should sum to 1."""
        _ = tree_layer(sample_input)
        mu = tree_layer._path_probabilities(sample_input)
        row_sums = tf.reduce_sum(mu, axis=1)
        tf.debugging.assert_near(row_sums, tf.ones_like(row_sums), atol=1e-6)


class TestDepth1Formula:
    """Tests for depth-1 manual formula verification."""

    def test_matches_manual_formula(self, depth1_layer, depth1_input):
        """Output should match 0.5*pi0 + 0.5*pi1 formula."""
        _ = depth1_layer(depth1_input)

        depth1_layer.pi.assign(tf.constant([[10.0], [20.0]], dtype=tf.float32))

        decision_dense = depth1_layer.decision
        decision_dense.set_weights(
            [
                tf.zeros_like(decision_dense.get_weights()[0]),
                tf.zeros_like(decision_dense.get_weights()[1]),
            ]
        )

        y = depth1_layer(depth1_input)
        expected = 0.5 * 10.0 + 0.5 * 20.0

        tf.debugging.assert_near(y, tf.fill((2, 1), expected), atol=1e-6)


class TestFeatureSample:
    """Tests for feature sampling behavior."""

    def test_never_zero_features(
        self, small_feature_sample_layer, feature_sample_input
    ):
        """Very small feature_sample should still keep at least one feature."""
        _ = small_feature_sample_layer(feature_sample_input)
        mask = small_feature_sample_layer.features_mask
        assert int(mask.shape[0]) >= 1
        assert int(mask.shape[1]) == 10
