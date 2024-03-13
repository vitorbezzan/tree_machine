"""
Config file for the trees module.
"""
# fmt: off
from functools import partial

from sklearn.metrics import (f1_score, mean_absolute_error,  # type: ignore
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             precision_score, recall_score)

# fmt: on

# Default hyperparams space for bayesian search
default_hyperparams = {
    "n_estimators": (2, 200),
    "max_depth": (2, 6),
}

regression_metrics = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "median": median_absolute_error,
    "mape": mean_absolute_percentage_error,
}

classification_metrics = {
    # Precision for binary classification
    "precision": precision_score,
    # Precision for multiclass classification
    "precision_micro": partial(precision_score, average="micro"),
    "precision_macro": partial(precision_score, average="macro"),
    "precision_weighted": partial(precision_score, average="weighted"),
    "precision_samples": partial(precision_score, average="samples"),
    # Recall for binary classification
    "recall": recall_score,
    # Recall for multiclass classification
    "recall_micro": partial(recall_score, average="micro"),
    "recall_macro": partial(recall_score, average="macro"),
    "recall_weighted": partial(recall_score, average="weighted"),
    "recall_samples": partial(recall_score, average="samples"),
    # F1 score for binary classification
    "f1": f1_score,
    # F1 score for multiclass classification
    "f1_micro": partial(f1_score, average="micro"),
    "f1_macro": partial(f1_score, average="macro"),
    "f1_weighted": partial(f1_score, average="weighted"),
    "f1_samples": partial(f1_score, average="samples"),
}
