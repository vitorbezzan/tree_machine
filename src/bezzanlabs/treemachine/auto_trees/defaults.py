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

_FloatLike = tp.Union[tuple[float, float], tp.Sequence[float]]
_IntLike = tp.Union[tuple[int, int], tp.Sequence[int]]


class TUsrDistribution(TypedDict, total=False):
    """
    Defines acceptable hyperparameters and their respective types for bounds when
        searching for the best model.

    Please see https://xgboost.readthedocs.io/en/latest/parameter.html for more details
        on these parameters work in your model.
    """

    eta: _FloatLike
    gamma: _FloatLike
    reg_alpha: _FloatLike
    colsample_bytree: _FloatLike
    colsample_bylevel: _FloatLike
    colsample_bynode: _FloatLike
    reg_lambda: _FloatLike
    max_depth: _IntLike
    n_estimators: _IntLike


defaults: TUsrDistribution = {
    "reg_alpha": (0.0, 1000.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (0.0, 1000.0),
    "max_depth": (2, 6),
    "n_estimators": (1, 1000),
}


@validate_call(config={"arbitrary_types_allowed": True})
def get_param_distributions(user_params: TUsrDistribution) -> TDistribution:
    """
    Returns distribution dictionary for parameters given user input.
    """
    params_: TDistribution = {}
    for param_name, domain in user_params.items():
        if isinstance(domain, tp.Sequence):
            params_[param_name] = CategoricalDistribution(domain)
        elif isinstance(domain, tuple):
            if isinstance(domain[0], int):
                params_[param_name] = IntDistribution(domain[0], int(domain[1]))
            else:
                params_[param_name] = FloatDistribution(domain[0], float(domain[1]))

    return params_
