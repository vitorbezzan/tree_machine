# fmt: off

"""
Some type definitions for the tree submodule.
"""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as imblearn_pipe  # type: ignore
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline as scikit_pipe  # type: ignore

Inputs = NDArray[np.float64] | pd.DataFrame
Actuals = NDArray[np.float64] | pd.DataFrame | pd.Series
Predictions = NDArray[np.float64]
Pipe = scikit_pipe | imblearn_pipe

# fmt: on
