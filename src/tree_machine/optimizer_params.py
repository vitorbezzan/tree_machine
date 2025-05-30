"""
Minimal configuration for hyperparameter distributions in optimization.
"""

from optuna import Trial


class OptimizerParams:
    """
    Defines acceptable hyperparameters and their respective types for bounds when
    searching for the best tree model.

    Please see https://xgboost.readthedocs.io/en/latest/parameter.html for more details
    on these parameters work in your model, if you are using trees.
    """

    hyperparams_grid = {
        "eta": (0.1, 0.6, 0.01),
        "gamma": (0.01, 0.6, 0.01),
        "reg_alpha": (0.0, 1000.0, 10.0),
        "colsample_bytree": (0.1, 1.0, 0.01),
        "colsample_bylevel": (0.1, 1.0, 0.01),
        "colsample_bynode": (0.1, 1.0, 0.01),
        "reg_lambda": (0.0, 1000.0, 10.0),
        "max_depth": (2, 10),
        "n_estimators": (1, 2000),
    }

    def get_trial_values(self, trial: Trial) -> dict:
        """
        Returns optuna trial values for functions.
        """
        values = {}
        for parameter, limit in self.hyperparams_grid.items():
            if isinstance(limit, tuple):
                if isinstance(limit[0], float):
                    values[parameter] = trial.suggest_float(
                        parameter,
                        limit[0],
                        limit[1],
                        step=limit[2] if len(limit) == 3 else None,
                    )
                elif isinstance(limit[0], int):
                    values[parameter] = trial.suggest_int(
                        parameter,
                        limit[0],
                        limit[1],  # type: ignore
                    )
            elif isinstance(limit, list):
                values[parameter] = trial.suggest_categorical(parameter, limit)
            else:
                raise RuntimeError(f"Parameter {parameter} format not recognized.")

        return values


class BalancedParams(OptimizerParams):
    """
    Set of balanced params to work with regularization factors for tree models.
    """

    hyperparams_grid = {
        "reg_alpha": (0.0, 5000.0, 10.0),
        "reg_lambda": (0.0, 5000.0, 10.0),
        "max_depth": (2, 10),
        "n_estimators": (1, 2000),
    }
