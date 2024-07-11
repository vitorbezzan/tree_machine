# isort: skip_file
"""
Minimal configuration file for Auto trees.
"""
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from pydantic import validate_call


defaults = {
    "alpha": FloatDistribution(0.0, 1000),
    "colsample_bytree": FloatDistribution(0.5, 1.0),
    "lambda": FloatDistribution(0.0, 1000),
    "max_depth": IntDistribution(2, 6),
    "n_estimators": IntDistribution(1, 1000),
}


_input_dict = dict[str, tuple[int, int] | tuple[float, float] | list]
_dist_dict = dict[str, CategoricalDistribution | IntDistribution | FloatDistribution]


@validate_call(config=dict(arbitrary_types_allowed=True))
def get_param_distributions(params: _input_dict) -> _dist_dict:
    """
    Returns distribution dictionary for parameters given user input.
    """
    params_: _dist_dict = {}
    for param_name, domain in params.items():
        if isinstance(domain, list):
            params_[param_name] = CategoricalDistribution(domain)
        elif isinstance(domain[0], int):
            params_[param_name] = IntDistribution(domain[0], int(domain[1]))
        else:
            params_[param_name] = FloatDistribution(domain[0], float(domain[1]))

    return params_
