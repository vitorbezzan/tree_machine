# flake8: noqa
"""
API entrypoint for the package.
"""

__package_name__ = "bezzanlabs.treemachine"
__version__ = "1.1.2"

from .auto_trees.base import BaseAuto
from .auto_trees.classifier import Classifier
from .auto_trees.regressor import Regressor
from .deep_trees.base import BaseDeep
from .deep_trees.classifier import DeepTreeClassifier
from .deep_trees.regressor import DeepTreeRegressor
from .splitter_proto import SplitterLike
