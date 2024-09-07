#!/usr/bin/env python3
import numpy as np


def getitem(obj, keys):
    """
    Get the item from the nested dictionary.
    """
    for key in keys:
        obj = obj.get(key, {})
    return obj


def scalar2array(obj, size, n1=None, n2=None):
    """
    Convert the scalar to an array.
    """
    if hasattr(obj, "__len__"):
        if len(obj) != size:
            if len(obj) == 1:
                return np.asarray([obj[0]] * size)
            if n1 is not None and n2 is not None:
                msg = f"The length of {n1} is not equal to {n2}"
            else:
                msg = f"The length of obj is not equal to {size}"
            raise ValueError(msg)
        return np.asarray(obj)
    else:
        return np.asarray([obj] * size)


def asarray_or_eval(array):
    """
    Convert the 'array' to an array by eval if it is a string.
    """
    if isinstance(array, str):
        return eval(array)
    else:
        z = np.asarray(array)
        if z.ndim == 0:
            return np.atleast_1d(z)
        else:
            return z
