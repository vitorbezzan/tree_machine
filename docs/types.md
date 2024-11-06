# types.py

## Summary

This file defines custom type aliases for input data, ground truth, and predictions used in the package.

## Dependencies

### Standard Library
- None

### Other
- numpy
- pandas

## Description

The `types.py` file provides type definitions that are used throughout the package to ensure type consistency and improve code readability. It defines three main type aliases:

1. `Inputs`: Represents input data, which can be either a NumPy array of float64 values or a pandas DataFrame.

2. `GroundTruth`: Represents ground truth data, which can be either a NumPy array of float64 values or a pandas Series.

3. `Predictions`: Represents prediction data, which is defined as a NumPy array of float64 values.

These type aliases are particularly useful in machine learning and data analysis contexts, where input data, ground truth labels, and model predictions are common entities. By using these type aliases, the package can maintain consistent type hints across different modules and functions, enhancing code clarity and facilitating static type checking.

*This documentation was generated using claude-3-5-sonnet-20240620*
