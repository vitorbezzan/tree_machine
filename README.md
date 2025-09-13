# Tree Machine: An AutoML companion to fit tree models easily

[![python](https://img.shields.io/badge/python-3.12-blue?style=for-the-badge)](http://python.org)
[![python](https://img.shields.io/badge/python-3.13-blue?style=for-the-badge)](http://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

This package aims to give users a simple interface to fit tree models. Tree models are
the workhorse for tabular data and are used in many applications. Our aim is to simplify
the use, tuning and deployment of these models.

### AutoTrees
Specific auto-tune trees that use Bayesian optimization to select the best model overall
and the best hyperparameters for that model. The models are trained using a `xgboost`
backend, and the user can change the parameters to use during `fit`.

Can be used as a last step inside a `sklearn.pipeline` object.

## Installing the package

Just issue the command:

```bash
python -m pip install git+https://github.com/vitorbezzan/tree_machine.git@main
```

This package should work in all systems, and it was tested in Linux and MacOS.

## Setup for development

To install this package, run the following command in your terminal:
```bash
make install
```
This should proceed with the installation of all dependencies for development.
