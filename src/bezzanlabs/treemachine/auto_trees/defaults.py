# isort: skip_file
"""
Minimal configuration file for Auto trees.
"""
import typing as tp
from typing_extensions import TypedDict
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from pydantic import validate_call

TDistribution: tp.TypeAlias = dict[
    str, CategoricalDistribution | IntDistribution | FloatDistribution
]

Float = tp.Union[tuple[float, float], list[float]]
Int = tp.Union[tuple[int, int], list[int]]


class TUsrDistribution(TypedDict, total=False):
    """
    Defines acceptable hyperparameters and their respective types for bounds when
        searching for the best model.

    Please see https://xgboost.readthedocs.io/en/latest/parameter.html for more details
        on these parameters work in your model.
    """

    eta: Float
    gamma: Float
    reg_alpha: Float
    colsample_bytree: Float
    colsample_bylevel: Float
    colsample_bynode: Float
    reg_lambda: Float
    max_depth: Int
    n_estimators: Float


defaults: TUsrDistribution = {
    "eta": (0.1, 0.6),
    "gamma": (0.0, 0.6),
    "reg_alpha": (0.0, 1000.0),
    "colsample_bytree": (0.5, 1.0),
    "colsample_bylevel": (0.5, 1.0),
    "colsample_bynode": (0.5, 1.0),
    "reg_lambda": (0.0, 1000.0),
    "max_depth": (2, 10),
    "n_estimators": (1, 1000),
}


@validate_call(config={"arbitrary_types_allowed": True})
def get_param_distributions(user_params: TUsrDistribution) -> TDistribution:
    """
    Returns distribution dictionary for parameters given user input.
    """
    params_: TDistribution = {}
    for param_name, domain in user_params.items():
        if isinstance(domain, list):
            params_[param_name] = CategoricalDistribution(domain)
        elif isinstance(domain, tuple):
            if isinstance(domain[0], int):
                params_[param_name] = IntDistribution(domain[0], int(domain[1]))
            else:
                params_[param_name] = FloatDistribution(domain[0], float(domain[1]))

    return params_
