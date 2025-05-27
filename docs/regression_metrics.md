# regression_metrics.py

## Summary
This code defines and validates regression metrics for machine learning evaluation.

## Dependencies

### Standard Library
- typing_extensions

### Other
- sklearn.metrics
- pydantic

## Description

This Python module provides a collection of commonly used regression metrics for evaluating machine learning models. It imports specific metric functions from scikit-learn's metrics module and organizes them into a dictionary for easy access and use.

The `regression_metrics` dictionary maps string keys to their corresponding metric functions. The available metrics include Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Median Absolute Error, Mean Squared Error (MSE), and Mean Pinball Loss (for quantile regression).

To ensure that only valid regression metrics are used, the module defines a custom validation function `_is_regression_metric`. This function checks if a given metric name is present in the `regression_metrics` dictionary and raises an assertion error if it's not.

The module also utilizes Pydantic's `AfterValidator` to create an `AcceptableRegression` type. This type annotation can be used to validate input parameters in other parts of the codebase, ensuring that only valid regression metric names are accepted.

