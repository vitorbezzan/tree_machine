# isort: skip_file
"""
All metrics available for regression.
"""

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    mean_pinball_loss,
)
from typing_extensions import Annotated
from pydantic import AfterValidator
from .types import Metric


regression_metrics = {
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "median": median_absolute_error,
    "mse": mean_squared_error,
    "quantile": mean_pinball_loss,
}


def _is_regression_metric(metric: Metric) -> Metric:
    """
    Validates that a metric is either a valid predefined regression metric
    or a callable custom metric function.
    """
    if callable(metric):
        return metric
    elif isinstance(metric, str):
        if metric not in regression_metrics:
            available_metrics = ", ".join(regression_metrics.keys())
            raise ValueError(
                f"Unknown regression metric '{metric}'. Available predefined metrics: "
                f"{available_metrics}. You can also pass a custom metric function."
            )
        return metric
    else:
        raise ValueError(
            f"Regression metric must be either a string or callable, got {type(metric)}"
        )


type AcceptableRegression = Annotated[Metric, AfterValidator(_is_regression_metric)]
