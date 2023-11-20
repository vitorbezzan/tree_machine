# fmt: off

"""
Some type definitions for the tree submodule.
"""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as imblearn_pipe  # type: ignore
from numpy.typing import NDArray
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline as scikit_pipe  # type: ignore

Inputs = NDArray[np.float64] | pd.DataFrame
Actuals = NDArray[np.float64] | pd.DataFrame
Predictions = NDArray[np.float64]
Pipe = scikit_pipe | imblearn_pipe

regression_metrics = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "median": median_absolute_error,
    "mape":  mean_absolute_percentage_error,
}

classification_metrics = {
    "auc": roc_auc_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

# fmt: on
