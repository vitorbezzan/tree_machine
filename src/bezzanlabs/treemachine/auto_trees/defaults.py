# isort: skip_file
"""
Minimal configuration file for Auto trees.
"""
from typing import TypeAlias
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from pydantic import validate_call


TUsrDistribution: TypeAlias = dict[str, tuple[int, int] | tuple[float, float] | list]
TDistribution: TypeAlias = dict[
    str, CategoricalDistribution | IntDistribution | FloatDistribution
]

defaults: TUsrDistribution = {
    "alpha": (0.0, 1000.0),
    "colsample_bytree": (0.5, 1.0),
    "lambda": (0.0, 1000.0),
    "max_depth": (2, 6),
    "n_estimators": (1, 1000),
}


@validate_call(config={"arbitrary_types_allowed": True})
def get_param_distributions(params: TUsrDistribution) -> TDistribution:
    """
    Returns distribution dictionary for parameters given user input.
    """
    params_: TDistribution = {}
    for param_name, domain in params.items():
        if isinstance(domain, list):
            params_[param_name] = CategoricalDistribution(domain)
        elif isinstance(domain[0], int):
            params_[param_name] = IntDistribution(domain[0], int(domain[1]))
        else:
            params_[param_name] = FloatDistribution(domain[0], float(domain[1]))

    return params_
