"""
No-split generator.
"""
import typing as tp
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..trees.types import Actuals


class DatetimeSplit(object):
    """
    CV-generator for time series data. Creates splits based in list of datetimes.

    This CV generator needs pd.DataFrames to work.
    """

    def __init__(self, date_list: tp.List[datetime]):
        """
        Constructor for CV-helper DatetimeSplit.

        Args:
            date_list: List containing datetimes to use as splits.
        """
        self._date_list = date_list

    def get_n_splits(
        self,
        X: pd.DataFrame,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> int:
        """
        Gets number of splits in validator. Always 1.
        """
        return len(self._date_list)

    def split(
        self,
        X: pd.DataFrame,
        y: Actuals | None = None,
        groups: NDArray[np.float64] | None = None,
    ) -> tp.Iterable[tp.Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Make consecutive splits using data given in X

        Yields:
            int: yields ranges for splitting
        """
        for i, value in enumerate(self._date_list):
            validation_start = X.index.get_loc(value)
            if i + 1 == len(self._date_list):
                validation_end = len(X)
            else:
                validation_end = X.index.get_loc(self._date_list[i + 1])

            yield (
                np.array(range(0, validation_start)),
                np.array(range(validation_start, validation_end)),
            )
