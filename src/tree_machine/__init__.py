# flake8: noqa
"""
API entrypoint for the package.
"""

__package_name__ = "bezzanlabs.tree_machine"
__version__ = "2.0.0"

from .base import BaseAutoCV, TExplainerResultsBase
from .regression_cv import RegressionCV, RegressionCVConfig, default_regression
