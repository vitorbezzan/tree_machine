"""
No-split generator.
"""
import typing as tp

import numpy as np
from numpy.typing import NDArray

from ..trees.types import Actuals, Inputs


class NoSplit(object):
    """
    Performs no-validation, i.e. returns same input.
    """

    def get_n_splits(
        self,
        X: Inputs,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> int:
        """
        Gets number of splits in validator. Always 1.
        """
        return 1

    def split(
        self,
        X: Inputs,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> tp.Iterable[tp.Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Make consecutive splits using data given in X
        """
        samples = X.shape[0]
        yield np.array(range(0, samples)), np.array(range(0, samples))
