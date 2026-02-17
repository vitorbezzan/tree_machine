"""Unit tests for the Keras DeepForest layer."""

import pickle

import pytest

tf = pytest.importorskip("tensorflow")
from tree_machine.deep_trees.layers import DeepForest  # noqa


@pytest.fixture(scope="session")
def forest_layer():
    """Standard DeepForest layer."""
    return DeepForest(n_trees=3, depth=2, feature_sample=0.7, output_size=4)


@pytest.fixture(scope="session")
def depth1_forest():
    """Depth-1 DeepForest for formula tests."""
    return DeepForest(n_trees=2, depth=1, feature_sample=1.0, output_size=1)


@pytest.fixture(scope="session")
def small_feature_forest():
    """DeepForest with very small feature sample."""
    return DeepForest(n_trees=4, depth=2, feature_sample=1e-9, output_size=1)


@pytest.fixture(scope="session")
def config_test_forest():
    """DeepForest for config roundtrip tests."""
    return DeepForest(n_trees=2, depth=3, feature_sample=0.5, output_size=2, name="df")


@pytest.fixture(scope="session")
def sample_input():
    """Sample input tensor."""
    return tf.random.normal((5, 10), seed=42)


@pytest.fixture(scope="session")
def depth1_input():
    """Input for depth-1 tests."""
    return tf.constant([[0.0, 0.0], [1.0, -1.0]], dtype=tf.float32)


@pytest.fixture(scope="session")
def feature_sample_input():
    """Input for feature sample tests."""
    return tf.random.normal((3, 10), seed=42)


@pytest.fixture(scope="session")
def config_input():
    """Input for config tests."""
    return tf.random.normal((2, 7), seed=42)


def _force_depth1_tree_constant_output(tree, leaf0: float, leaf1: float) -> None:
    """Force a depth-1 tree to output a deterministic constant."""
    tree.pi.assign(tf.constant([[leaf0], [leaf1]], dtype=tf.float32))
    decision_dense = tree.decision
    decision_dense.set_weights(
        [
            tf.zeros_like(decision_dense.get_weights()[0]),
            tf.zeros_like(decision_dense.get_weights()[1]),
        ]
    )


class TestOutputShape:
    """Tests for output shape and dtype."""

    def test_shape(self, forest_layer, sample_input):
        """Output shape is (n_samples, output_size)."""
        y = forest_layer(sample_input)
        assert tuple(y.shape) == (5, 4)

    def test_dtype(self, forest_layer, sample_input):
        """Output dtype is float32."""
        y = forest_layer(sample_input)
        assert y.dtype == tf.float32

    def test_finite(self, forest_layer, sample_input):
        """Output contains no non-finite values."""
        y = forest_layer(sample_input)
        tf.debugging.assert_all_finite(
            y, "DeepForest output contains non-finite values"
        )


class TestDepth1Formula:
    """Tests for depth-1 sum of trees formula."""

    def test_matches_manual_formula(self, depth1_forest, depth1_input):
        """Forced 0.5 split should match expected sum."""
        _ = depth1_forest(depth1_input)

        _force_depth1_tree_constant_output(
            depth1_forest.ensemble[0], leaf0=10.0, leaf1=20.0
        )
        _force_depth1_tree_constant_output(
            depth1_forest.ensemble[1], leaf0=1.0, leaf1=3.0
        )

        y = depth1_forest(depth1_input)
        expected = 15.0 + 2.0

        tf.debugging.assert_near(y, tf.fill((2, 1), expected), atol=1e-6)


class TestFeatureSample:
    """Tests for feature sampling behavior."""

    def test_never_zero_features(self, small_feature_forest, feature_sample_input):
        """Each tree should keep at least one feature."""
        _ = small_feature_forest(feature_sample_input)

        for tree in small_feature_forest.ensemble:
            mask = tree.features_mask
            assert int(mask.shape[0]) >= 1
            assert int(mask.shape[1]) == 10


class TestConfigRoundtrip:
    """Tests for config serialization."""

    def test_config_values_preserved(self, config_test_forest):
        """Config values are preserved through roundtrip."""
        config = config_test_forest.get_config()
        cloned = DeepForest.from_config(config)

        assert cloned.n_trees == config_test_forest.n_trees
        assert cloned.depth == config_test_forest.depth
        assert cloned.feature_sample == config_test_forest.feature_sample
        assert cloned.output_size == config_test_forest.output_size

    def test_cloned_is_callable(self, config_test_forest, config_input):
        """Cloned layer is callable with correct output shape."""
        config = config_test_forest.get_config()
        cloned = DeepForest.from_config(config)

        y = cloned(config_input)
        assert tuple(y.shape) == (2, 2)


class TestPickleRoundtrip:
    """Tests for pickle serialization."""

    def test_preserves_outputs(self, depth1_forest, depth1_input):
        """Pickled/unpickled layer preserves outputs."""
        _ = depth1_forest(depth1_input)

        _force_depth1_tree_constant_output(
            depth1_forest.ensemble[0], leaf0=10.0, leaf1=20.0
        )
        _force_depth1_tree_constant_output(
            depth1_forest.ensemble[1], leaf0=1.0, leaf1=3.0
        )

        before = depth1_forest(depth1_input)

        pickled = pickle.dumps(depth1_forest)
        restored = pickle.loads(pickled)

        after = restored(depth1_input)

        tf.debugging.assert_near(before, after, atol=1e-6)
