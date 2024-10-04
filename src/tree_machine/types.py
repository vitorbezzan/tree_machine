# fmt: off

"""
Some type definitions for the package.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

Inputs = NDArray[np.float64] | pd.DataFrame
GroundTruth = NDArray[np.float64] | pd.Series
Predictions = NDArray[np.float64]

# fmt: on
