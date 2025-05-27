# classifier_cv.py

## Summary

This code defines a `ClassifierCV` class for automated classification using XGBoost with Bayesian optimization and SHAP explanations.

## Dependencies

### Standard Library
- typing
- multiprocessing

### Other
- numpy
- pandas
- pydantic
- sklearn
- xgboost
- shap (optional)

## Description

The `classifier_cv.py` file implements an automated classification system using XGBoost as the base classifier. The core components include:

1. `ClassifierCVConfig`: A Pydantic dataclass that defines configuration options for the classifier, including:
   - `monotone_constraints`: Dictionary specifying monotonicity direction for variables (0 for none, 1 for increasing, -1 for decreasing)
   - `interactions`: List of lists containing permitted feature interactions
   - `n_jobs`: Number of parallel jobs to use when fitting the model
   - `parameters`: Hyperparameter search space definition (OptimizerParams instance)
   - `return_train_score`: Whether to include training scores during optimization

2. Pre-configured settings:
   - `default_classifier`: A standard configuration using all hyperparameters
   - `balanced_classifier`: A configuration focused on regularization parameters

3. `ClassifierCV`: The main class that inherits from `BaseAutoCV`, `ClassifierMixin`, and `ExplainerMixIn`, providing:
   - Automated hyperparameter tuning via Bayesian optimization
   - XGBoost-based classification capabilities
   - Model explanation using SHAP values (when available)
   - Feature importance calculation
   - Parallel processing support

The implementation gracefully handles cases where the optional SHAP library isn't available, still allowing the core classification functionality to work while disabling the explanation features.

The class supports custom classification metrics through integration with scikit-learn's scoring system and Pydantic's validation mechanism, ensuring that only valid metrics are used. Cross-validation is employed during the optimization process to prevent overfitting.

Overall, this module provides a robust, automated approach to building classification models with XGBoost, combining the power of Bayesian optimization for parameter tuning with the interpretability offered by SHAP explanations.
