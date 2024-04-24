"""
This module defines radial kernel functions.
"""
import numpy as np
from sklearn.model_selection import ParameterGrid
from functools import partial


filter_presets = {
    "pexp": {
        "small": {
            "rl": np.linspace(1.0, 4.0, num=8, endpoint=True),
            "pl": np.linspace(3.0, 1.0, num=8, endpoint=True)
        },
        "medium": {
            "rl": np.linspace(1.0, 4.0, num=16, endpoint=True),
            "pl": np.linspace(3.0, 1.0, num=16, endpoint=True)
        },
        "large": {
            "rl": np.linspace(1.0, 4.0, num=32, endpoint=True),
            "pl": np.linspace(3.0, 1.0, num=32, endpoint=True)
        }
    },
    "morse": {
        "small": {
            "D": np.ones(1, dtype=float),
            "r0": np.linspace(1.4, 3.2, num=10, endpoint=True),
            "gamma": np.ones(1, dtype=float)
        },
        "medium": {
            "D": np.ones(1, dtype=float),
            "r0": np.linspace(1.4, 3.2, num=10, endpoint=True),
            "gamma": np.array([1.0, 2.0])
        },
        "large": {
            "D": np.array([0.8, 1.2]),
            "r0": np.linspace(1.4, 3.2, num=10, endpoint=True),
            "gamma": np.array([1.0, 2.0])
        }
    }
}


def pexp(x: np.ndarray, r: float, p: float):
    """
    The parameterized exponential function.
    """
    f = np.exp(-(x / r) ** p)
    df = -f * p * (x / r) ** (p - 1.0) / r
    return f, df


def morse(x: np.ndarray, D, gamma, r0):
    """
    The morse function.
    f(x) = d * [ exp(-2 * gamma * (r - r0)) - 2 * exp(-gamma * (r - r0)) ]
    """
    dr = gamma * (x - r0)
    f = D * (np.exp(-2 * dr) - 2 * np.exp(-dr))
    df = -D * gamma * 2 * (np.exp(-2 * dr) - np.exp(-dr))
    return f, df


def fcut(x: np.ndarray, cutforce: float):
    """
    The cosine cutoff function.
    """
    z = np.minimum(1.0, x / cutforce) * np.pi
    f = 0.5 * (np.cos(z) + 1)
    df = -0.5 * np.pi / cutforce * np.sin(z)
    return f, df


def get_filter_preset(key: str):
    """
    Return a pre-defined collection of filter functions.
    """
    vals = key.split("@")
    if len(vals) != 2:
        raise KeyError(f"{key} is not valid preset. Should be func@size")
    func = vals[0]
    size = vals[1]
    if func == "pexp":
        params = filter_presets[func][size]
        filters = [partial(pexp, r=params['rl'][i], p=params['pl'][i])
                   for i in range(len(params['rl']))]
    elif func == "morse":
        params = filter_presets[func][size]
        grid = ParameterGrid(params)
        filters = []
        for row in grid:
            D = row["D"]
            r0 = row["r0"]
            gamma = row["gamma"]
            filters.append(partial(morse, D=D, r0=r0, gamma=gamma))
    else:
        raise KeyError("Only morse or pexp functions are supported")
    return filters
