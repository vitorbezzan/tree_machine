# isort: skip_file
"""
All metrics available for classification.
"""

from functools import partial
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from typing_extensions import Annotated
from pydantic import AfterValidator
from .types import Metric


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


def _is_classification_metric(metric: Metric) -> Metric:
    """
    Validates that a metric is either a valid predefined classification metric
    or a callable custom metric function.
    """
    if callable(metric):
        return metric
    elif isinstance(metric, str):
        if metric not in classification_metrics:
            available_metrics = ", ".join(classification_metrics.keys())
            raise ValueError(
                f"Unknown classification metric '{metric}'. Available predefined metrics: "
                f"{available_metrics}. You can also pass a custom metric function."
            )
        return metric
    else:
        raise ValueError(
            f"Classification metric must be either a string or callable, got {type(metric)}"
        )


type AcceptableClassifier = Annotated[Metric, AfterValidator(_is_classification_metric)]
