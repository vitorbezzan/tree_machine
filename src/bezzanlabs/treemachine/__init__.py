# flake8: noqa
"""
API entrypoint for the package.
"""
from .cv.datetime import DatetimeSplit
from .cv.no_split import NoSplit
from .trees.base import BaseTree, SplitterLike
from .trees.classifier import ClassifierTree
from .trees.regressor import RegressorTree

__package_name__ = "treemachine"
__version__ = "0.0.1"
