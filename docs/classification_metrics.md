# classification_metrics.py

## Summary
This code defines a collection of classification metrics and provides a validation mechanism for acceptable classifier names.

## Dependencies

### Standard Library
- functools
- typing_extensions

### Other
- sklearn.metrics
- pydantic

## Description

This Python module provides a comprehensive set of classification metrics commonly used in machine learning tasks. It leverages the scikit-learn library to access various scoring functions such as F1 score, precision, and recall.

The `classification_metrics` dictionary is the core of this module. It maps human-readable metric names to their corresponding scikit-learn functions. For each metric type (F1, precision, recall), it includes both the default version and variants with different averaging methods (macro, micro, samples, weighted). This allows for easy access to a wide range of classification evaluation metrics.

The module also defines a custom validation function `_is_classification_metric` which checks if a given metric name is present in the `classification_metrics` dictionary. This function is used in conjunction with Pydantic's `AfterValidator` to create an `AcceptableClassifier` type. This type can be used to ensure that only valid classification metric names are accepted in other parts of the codebase that might use this module.

By providing a centralized collection of classification metrics and a validation mechanism, this module promotes consistency and error prevention when working with classification tasks in a larger machine learning project.

*This documentation was generated using claude-3-5-sonnet-20240620*
