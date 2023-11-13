# fmt: off

"""
Some type definitions for the tree submodule.
"""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as imblearn_pipe  # type: ignore
from numpy.typing import NDArray
from sklearn.metrics import (f1_score, make_scorer,  # type: ignore
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline as scikit_pipe  # type: ignore

Inputs = NDArray[np.float64] | pd.DataFrame
Actuals = NDArray[np.float64] | pd.DataFrame
Predictions = NDArray[np.float64]
Pipe = scikit_pipe | imblearn_pipe

regression_metrics = {
    "mse": make_scorer(mean_squared_error, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "median": make_scorer(median_absolute_error, greater_is_better=False),
    "mape":  make_scorer(mean_absolute_percentage_error, greater_is_better=False),
}

classification_metrics = {
    "auc": make_scorer(roc_auc_score, needs_proba=True),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
}

# fmt: on
