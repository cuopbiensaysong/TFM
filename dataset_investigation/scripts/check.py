import pickle 
import pandas as pd
import numpy as np
import importlib

import sys

def _read_pickle_with_numpy_compat(path):
    """Load pickles across NumPy 1.x/2.x internal module path differences."""
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or ""
        if missing_module.startswith("numpy._core"):
            # NumPy 2.x pickle loaded under NumPy 1.x runtime.
            mapped_module = missing_module.replace("numpy._core", "numpy.core", 1)
            sys.modules[missing_module] = importlib.import_module(mapped_module)
            return pd.read_pickle(path)
        if missing_module.startswith("numpy.core"):
            # NumPy 1.x pickle loaded under NumPy 2.x runtime.
            mapped_module = missing_module.replace("numpy.core", "numpy._core", 1)
            sys.modules[missing_module] = importlib.import_module(mapped_module)
            return pd.read_pickle(path)
        raise


data = _read_pickle_with_numpy_compat("dataset_investigation/outputs/eicu_sepsis_eicu.pkl")

print(data.keys())
print(data["train"].head())
print(data["val"].head())
print(data["test"].head())