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

    _xgboost_to_catboost_map = {
        "eta": "learning_rate",
        "gamma": "min_data_in_leaf",
        "reg_alpha": "l2_leaf_reg",
        "reg_lambda": "l2_leaf_reg",
        "colsample_bytree": "rsm",
        "max_depth": "max_depth",
        "n_estimators": "iterations",
    }

    _xgboost_to_lightgbm_map = {
        "eta": "learning_rate",
        "gamma": "min_split_gain",
        "reg_alpha": "lambda_l1",
        "reg_lambda": "lambda_l2",
        "colsample_bytree": "feature_fraction",
        "colsample_bylevel": "feature_fraction",
        "colsample_bynode": "feature_fraction",
        "max_depth": "max_depth",
        "n_estimators": "n_estimators",
    }

    def get_trial_values(self, trial: Trial, backend: str = "xgboost") -> dict:
        """
        Returns optuna trial values for functions.

        Args:
            trial: Optuna trial object.
            backend: Backend to use. Either "xgboost", "catboost" or "lightgbm".
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

        if backend == "catboost":
            return self._map_to_catboost(values)
        if backend == "lightgbm":
            return self._map_to_lightgbm(values)
        return values

    def _map_to_catboost(self, xgboost_params: dict) -> dict:
        """
        Maps XGBoost parameter names to CatBoost parameter names.

        Args:
            xgboost_params: Dictionary of XGBoost parameters.

        Returns:
            Dictionary of CatBoost parameters.
        """
        catboost_params = {}
        l2_leaf_reg_values = []

        for xgb_param, value in xgboost_params.items():
            if xgb_param in self._xgboost_to_catboost_map:
                catboost_param = self._xgboost_to_catboost_map[xgb_param]

                if catboost_param == "l2_leaf_reg":
                    l2_leaf_reg_values.append(value)
                elif catboost_param == "rsm":
                    if (
                        catboost_param not in catboost_params
                        or xgb_param == "colsample_bytree"
                    ):
                        catboost_params[catboost_param] = value
                elif catboost_param not in catboost_params:
                    catboost_params[catboost_param] = value

        if l2_leaf_reg_values:
            catboost_params["l2_leaf_reg"] = sum(l2_leaf_reg_values) / len(
                l2_leaf_reg_values
            )

        return catboost_params

    def _map_to_lightgbm(self, xgboost_params: dict) -> dict:
        """
        Maps XGBoost parameter names to LightGBM parameter names.

        Args:
            xgboost_params: Dictionary of XGBoost parameters.

        Returns:
            Dictionary of LightGBM parameters.
        """
        lightgbm_params = {}
        feature_fraction_values = []

        for xgb_param, value in xgboost_params.items():
            if xgb_param in self._xgboost_to_lightgbm_map:
                lgbm_param = self._xgboost_to_lightgbm_map[xgb_param]

                if lgbm_param == "feature_fraction":
                    feature_fraction_values.append(value)
                elif lgbm_param not in lightgbm_params:
                    lightgbm_params[lgbm_param] = value

        if feature_fraction_values:
            lightgbm_params["feature_fraction"] = sum(feature_fraction_values) / len(
                feature_fraction_values
            )

        return lightgbm_params


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
