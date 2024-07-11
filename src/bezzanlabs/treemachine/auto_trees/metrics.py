# isort: skip_file
"""
All metrics available for for
"""
from functools import partial
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
)

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
