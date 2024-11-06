# flake8: noqa
"""
API entrypoint for the package.
"""

__version__ = "2.0.0"

from .classifier_cv import ClassifierCV, ClassifierCVConfig, default_classifier
from .optimizer_params import OptimizerParams
from .regression_cv import RegressionCV, RegressionCVConfig, default_regression
