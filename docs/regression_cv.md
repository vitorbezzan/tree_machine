# regression_cv.py

## Summary
This code defines a RegressionCV class for automated regression tree modeling with pluggable gradient-boosting backends using Bayesian optimization.

## Dependencies

### Standard Library
- typing
- multiprocessing
- functools

### Other
- numpy
- pandas
- pydantic
- sklearn
- xgboost
- catboost
- lightgbm
- shap (optional)

## Description

The `regression_cv.py` file implements a `RegressionCV` class, which is an automated regression tree model based on Bayesian optimization. This class combines several machine learning techniques to create a powerful and flexible regression model.

The implementation includes:

1. `RegressionCVConfig`: A Pydantic dataclass that defines configuration options for the regressor, including:
   - `monotone_constraints`: Dictionary specifying monotonicity direction for variables (0 for none, 1 for increasing, -1 for decreasing)
   - `interactions`: List of lists containing permitted feature interactions
   - `n_jobs`: Number of parallel jobs to use when fitting the model
   - `parameters`: Hyperparameter search space definition (OptimizerParams instance)
   - `return_train_score`: Whether to include training scores during optimization
   - `quantile_alpha`: Optional parameter for quantile regression (specifies the quantile to predict)

2. Pre-configured settings:
   - `default_regression`: A standard configuration using all hyperparameters
   - `balanced_regression`: A configuration focused on regularization parameters

3. `RegressionCV`: The main class that inherits from `BaseAutoCV`, `RegressorMixin`, and `ExplainerMixIn`, providing:
   - Automated hyperparameter tuning via Bayesian optimization
   - Backend selection through `backend` (`"xgboost"`, `"catboost"`, or `"lightgbm"`)
   - Support for quantile regression when specified
   - Model explanation using SHAP values (when available)
   - Feature importance calculation
   - Parallel processing support

The implementation gracefully handles cases where the optional SHAP library isn't available, still allowing the core regression functionality to work while disabling the explanation features.

The class supports various regression metrics through integration with scikit-learn's scoring system and Pydantic's validation mechanism, ensuring that only valid metrics are used. Cross-validation is employed during the optimization process to prevent overfitting and ensure model robustness.

Overall, this module provides a comprehensive solution for regression tasks, offering automated model selection, support for advanced regression techniques like quantile regression, and model interpretability through SHAP explanations.
