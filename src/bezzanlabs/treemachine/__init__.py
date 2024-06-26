# flake8: noqa
"""
API entrypoint for the package.
"""

__package_name__ = "bezzanlabs.treemachine"
__version__ = "1.2.1"

from .auto_trees.base import BaseAutoTree
from .auto_trees.classifier_cv import ClassifierCV
from .auto_trees.regression_cv import RegressionCV
from .deep_trees.base import BaseDeep
from .deep_trees.deep_classifier import DeepTreeClassifier
from .deep_trees.deep_regressor import DeepTreeRegressor
