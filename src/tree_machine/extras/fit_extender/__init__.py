# flake8: noqa
"""
Extends the .fit() method in estimators to give them special functionality.
"""
from .fit_extender import IsEstimator, IsExtender, fit_extend
from .outlier_detector import OutlierDetector
from .train_data import TrainData
