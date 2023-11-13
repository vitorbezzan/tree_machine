# fmt: off
# flake8: noqa
"""
Monkey patching and fixes for some packages.
"""
import numpy as np
from skopt.space.transformers import Normalize  # type: ignore


def _inverse_transform(self, X):  # pragma: no cover
    X = np.asarray(X)
    if np.any(X > 1.0 + self._eps):
        raise ValueError("All values should be less than 1.0")
    if np.any(X < 0.0 - self._eps):
        raise ValueError("All values should be greater than 0.0")
    X_orig = X * (self.high - self.low) + self.low
    if self.is_int:
        return np.round(X_orig).astype(int)  # << Changed this line to `int`.
    return X_orig


def _transform(self, X):  # pragma: no cover
    X = np.asarray(X)
    if self.is_int:
        if np.any(np.round(X) > self.high):
            raise ValueError("All integer values should"
                             "be less than %f" % self.high)
        if np.any(np.round(X) < self.low):
            raise ValueError("All integer values should"
                             "be greater than %f" % self.low)
    else:
        if np.any(X > self.high + self._eps):
            raise ValueError("All values should"
                             "be less than %f" % self.high)
        if np.any(X < self.low - self._eps):
            raise ValueError("All values should"
                             "be greater than %f" % self.low)
    if (self.high - self.low) == 0.:
        return X * 0.
    if self.is_int:
        return (np.round(X).astype(int) - self.low) /\
               (self.high - self.low)  # << Changed this line to `int`.
    else:
        return (X - self.low) / (self.high - self.low)


def apply_patches():
    Normalize.inverse_transform = _inverse_transform
    Normalize.transform = _transform
