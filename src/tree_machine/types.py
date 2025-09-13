# fmt: off

"""
Some type definitions for the package.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

type Inputs = NDArray[np.float64] | pd.DataFrame
type GroundTruth = NDArray[np.float64] | pd.Series
type Predictions = NDArray[np.float64]

# fmt: on
