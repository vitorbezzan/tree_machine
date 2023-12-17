"""
Prototype for a splitter-like object.
"""
import typing as tp

import numpy as np
from numpy.typing import NDArray

from ..types import Actuals, Inputs


@tp.runtime_checkable
class SplitterLike(tp.Protocol):
    """
    Specifies a protocol for splitters. Defines the minimum specified behavior for these
    types of objects.
    """

    def get_n_splits(
        self,
        X: Inputs,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> int:
        """
        Returns the number of splits for a given dataset.
        """

    def split(
        self,
        X: Inputs,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> tp.Iterable[tp.Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Splits and yields data.
        """
