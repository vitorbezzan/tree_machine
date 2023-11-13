# flake8: noqa
"""
Trees module for the package. Implements base, regression and classification trees.
"""
from .base import BaseTree, SplitterLike
from .classifier import ClassifierTree
from .regressor import RegressorTree
