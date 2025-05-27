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


regression_metrics = {
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "median": median_absolute_error,
    "mse": mean_squared_error,
    "quantile": mean_pinball_loss,
}


def _is_regression_metric(metric: str) -> str:
    assert metric in regression_metrics
    return metric


AcceptableRegression = Annotated[str, AfterValidator(_is_regression_metric)]
