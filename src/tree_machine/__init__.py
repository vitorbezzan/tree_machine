# flake8: noqa
"""
API entrypoint for the package.
"""

__version__ = "202512.02"

from .classifier_cv import (
    ClassifierCV,
    ClassifierCVConfig,
    balanced_classifier,
    default_classifier,
)
from .optimizer_params import OptimizerParams
from .quantile_cv import QuantileCV
from .regression_cv import (
    RegressionCV,
    RegressionCVConfig,
    balanced_regression,
    default_regression,
)
