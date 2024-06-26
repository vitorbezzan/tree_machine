"""
Minimal configuration file for Deep auto trees.
"""

from optuna.distributions import FloatDistribution, IntDistribution

defaults = {
    "alpha_l1": FloatDistribution(0.0, 1000),
    "lambda_l2": FloatDistribution(0.0, 1000),
    "max_depth": IntDistribution(2, 6),
    "n_estimators": IntDistribution(2, 200),
    "internal_size": IntDistribution(4, 20),
}
