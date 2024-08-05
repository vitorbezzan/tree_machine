# metrics.py

## Summary

This code defines dictionaries of regression and classification metrics using scikit-learn functions.

## Dependencies

### Standard Library
- functools

### Other
- sklearn.metrics

## Description

The `metrics.py` file provides a collection of commonly used metrics for evaluating machine learning models, specifically for regression and classification tasks. It imports various metric functions from scikit-learn and organizes them into two dictionaries: `regression_metrics` and `classification_metrics`.

The `regression_metrics` dictionary contains four key metrics for evaluating regression models:
1. Mean Absolute Error (MAE)
2. Mean Absolute Percentage Error (MAPE)
3. Median Absolute Error
4. Mean Squared Error (MSE)

The `classification_metrics` dictionary is more extensive, offering a variety of metrics for classification tasks. It includes different variations of F1 score, precision, and recall, each with various averaging methods (macro, micro, samples, and weighted). The `partial` function from the `functools` module is used to create partially applied functions for metrics that require specific parameters.

This organization allows for easy access and use of these metrics in other parts of the codebase, providing a centralized location for metric definitions and promoting consistency in model evaluation across the project.

*This documentation was generated using claude-3-5-sonnet-20240620*