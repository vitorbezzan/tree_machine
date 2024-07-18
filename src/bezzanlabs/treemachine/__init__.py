# flake8: noqa
"""
API entrypoint for the package.
"""

__package_name__ = "bezzanlabs.treemachine"
__version__ = "1.2.3"

from .auto_trees.base import BaseAutoTree
from .auto_trees.classifier_cv import ClassifierCV
from .auto_trees.defaults import get_param_distributions
from .auto_trees.regression_cv import RegressionCV
