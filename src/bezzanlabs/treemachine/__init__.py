# flake8: noqa
"""
API entrypoint for the package.
"""
from .auto_trees.base import BaseAuto
from .auto_trees.classifier import Classifier
from .auto_trees.regressor import Regressor
from .auto_trees.splitter_proto import SplitterLike
from .deep_trees.base import BaseDeep
from .deep_trees.classifier import DeepTreeClassifier
from .deep_trees.regressor import DeepTreeRegressor

__package_name__ = "bezzanlabs.treemachine"
__version__ = "1.0.0"
