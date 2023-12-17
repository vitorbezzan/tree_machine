# flake8: noqa
"""
API entrypoint for the package.
"""
from .auto_trees.base import SplitterLike
from .auto_trees.classifier import Classifier
from .auto_trees.regressor import Regressor

__package_name__ = "bezzanlabs.treemachine"
__version__ = "0.0.1"
