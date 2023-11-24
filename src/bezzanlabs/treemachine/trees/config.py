"""
Config file for the trees module.
"""
# fmt: off
from sklearn.metrics import (f1_score, mean_absolute_error,  # type: ignore
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             precision_score, recall_score, roc_auc_score)

# fmt: on

# Default hyperparams space for bayesian search
default_hyperparams = {
    "n_estimators": (2, 100),
    "num_leaves": (20, 200),
}

regression_metrics = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "median": median_absolute_error,
    "mape": mean_absolute_percentage_error,
}

classification_metrics = {
    "auc": roc_auc_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}
