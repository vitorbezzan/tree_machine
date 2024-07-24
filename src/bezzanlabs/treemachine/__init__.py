# flake8: noqa
"""
API entrypoint for the package.
"""

__package_name__ = "bezzanlabs.treemachine"
__version__ = "1.2.4"

from .auto_trees.base import BaseAutoTree, TExplainerResults
from .auto_trees.classifier_cv import ClassifierCV, ClassifierCVOptions
from .auto_trees.regression_cv import RegressionCV, RegressionCVOptions
