# flake8: noqa
"""
API entrypoint for the package.
"""
from .trees.base import SplitterLike
from .trees.classifier import Classifier
from .trees.regressor import Regressor

__package_name__ = "treemachine"
__version__ = "0.0.1"
