# isort: skip_file
"""
Minimal configuration file for Auto trees.
"""
from functools import partial

from optuna.distributions import FloatDistribution, IntDistribution
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
)

defaults = {
    "alpha": FloatDistribution(0.0, 1000),
    "colsample_bytree": FloatDistribution(0.0, 1.0),
    "lambda": FloatDistribution(0.0, 1000),
    "max_depth": IntDistribution(2, 6),
    "n_estimators": IntDistribution(2, 200),
}

regression_metrics = {
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "median": median_absolute_error,
    "mse": mean_squared_error,
}

classification_metrics = {
    "f1": f1_score,
    "f1_macro": partial(f1_score, average="macro"),
    "f1_micro": partial(f1_score, average="micro"),
    "f1_samples": partial(f1_score, average="samples"),
    "f1_weighted": partial(f1_score, average="weighted"),
    "precision": precision_score,
    "precision_macro": partial(precision_score, average="macro"),
    "precision_micro": partial(precision_score, average="micro"),
    "precision_samples": partial(precision_score, average="samples"),
    "precision_weighted": partial(precision_score, average="weighted"),
    "recall": recall_score,
    "recall_macro": partial(recall_score, average="macro"),
    "recall_micro": partial(recall_score, average="micro"),
    "recall_samples": partial(recall_score, average="samples"),
    "recall_weighted": partial(recall_score, average="weighted"),
}
