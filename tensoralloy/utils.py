# coding=utf-8
"""
This module defines utility functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import logging

from dataclasses import fields
from genericpath import isdir
from os import makedirs
from os.path import dirname
from itertools import chain
from logging.config import dictConfig
from typing import List, Union, Any

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def add_slots(cls):
    """
    A decorator to put __slots__ on a dataclass with fields with defaults.

    Need to create a new class, since we can't set __slots__ after a class has
    been created.

    References
    ----------
    https://github.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py

    """
    # Make sure __slots__ isn't already set.
    if '__slots__' in cls.__dict__:
        raise TypeError(f'{cls.__name__} already specifies __slots__')

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in fields(cls))
    cls_dict['__slots__'] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop('__dict__', None)
    # And finally create the class.
    qualname = getattr(cls, '__qualname__', None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls


def cantor_pairing(x, y):
    """
    The Cantor Pairing function:

        f(x, y) = (x + y)(x + y + 1) // 2 + y

    f(x, y) will only be unique if x and y are integers.

    See Also
    --------
    https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function

    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert np.issubdtype(x.dtype, np.int_) and np.issubdtype(y.dtype, np.int_)
    return (x + y) * (x + y + 1) // 2 + y


def szudzik_pairing_scalar(x, y):
    """
    The szudzik pairing function for two scalars. This pairing function supports
    negative numbers.

    See Also
    --------
    https://gist.github.com/TheGreatRambler/048f4b38ca561e6566e0e0f6e71b7739

    """
    xx = x * 2 if x >= 0 else x * -2 - 1
    yy = y * 2 if y >= 0 else y * -2 - 1
    return xx * xx + xx + yy if xx >= yy else yy * yy + xx


def _szudzik_pairing(x, y):
    """
    The szudzik pairing function which supports negative numbers.

    See Also
    --------
    https://gist.github.com/TheGreatRambler/048f4b38ca561e6566e0e0f6e71b7739

    """
    # xx = x * 2 if x >= 0 else x * -2 - 1
    # yy = y * 2 if y >= 0 else y * -2 - 1
    # return xx * xx + xx + yy if xx >= yy else yy * yy + xx
    if np.isscalar(x):
        return szudzik_pairing_scalar(x, y)
    else:
        stack = []
        for v in (x, y):
            ind = v >= 0
            vv = np.zeros_like(v)
            vv[ind] = v[ind] * 2
            vv[~ind] = -2 * v[~ind] - 1
            stack.append(vv)
        xx, yy = stack
        ind = xx >= yy
        result = np.zeros_like(x)
        result[ind] = xx[ind] * xx[ind] + xx[ind] + yy[ind]
        result[~ind] = yy[~ind] * yy[~ind] + xx[~ind]
        return result


def szudzik_pairing(x, *args):
    """
    The szudzik pairing function.

    See Also
    --------
    https://gist.github.com/TheGreatRambler/048f4b38ca561e6566e0e0f6e71b7739

    """
    if np.isscalar(x):
        z = x
        for y in args:
            assert np.isscalar(y)
            z = _szudzik_pairing(z, y)
    else:
        x = np.asarray(x)
        if np.ndim(x) == 1:
            z = x
            for y in args:
                y = np.asarray(y)
                assert np.ndim(y) == 1
                z = _szudzik_pairing(z, y)
        elif np.ndim(x) == 2:
            z = x[:, 0]
            for col in range(1, x.shape[1]):
                z = _szudzik_pairing(z, x[:, col])
        else:
            raise ValueError("Dimension error")
    return z


def szudzik_pairing_reverse_scalar(z):
    """
    The reverse of `szudzik_pairing` for a scalar.
    """
    sqrtz = int(np.floor(np.sqrt(z)))
    sqz = sqrtz**2
    ab = (sqrtz, z - sqz - sqrtz) if (z - sqz) >= sqrtz else (z - sqz, sqrtz)
    xx = ab[0] // 2 if ab[0] % 2 == 0 else (ab[0] + 1) // -2
    yy = ab[1] // 2 if ab[1] % 2 == 0 else (ab[1] + 1) // -2
    return xx, yy


def szudzik_pairing_reverse(z):
    """
    The reverse of `szudzik_pairing`.
    """
    if np.isscalar(z):
        return szudzik_pairing_reverse_scalar(z)

    z = np.asarray(z)
    size = len(z)
    sqrtz = np.floor(np.sqrt(z)).astype(z.dtype)
    sqz = sqrtz**2
    diff = z - sqz
    ind = diff >= sqrtz

    ab = np.zeros((2, size), dtype=z.dtype)
    ab[0, ind] = sqrtz[ind]
    ab[1, ind] = z[ind] - sqz[ind] - sqrtz[ind]
    ab[0, ~ind] = z[~ind] - sqz[~ind]
    ab[1, ~ind] = sqrtz[~ind]

    xx = np.zeros_like(z)
    yy = np.zeros_like(z)

    ind = ab[0] % 2 == 0
    xx[ind] = ab[0, ind] // 2
    xx[~ind] = (ab[0, ~ind] + 1) // 2

    ind = ab[1] % 2 == 0
    yy[ind] = ab[1, ind] // 2
    yy[~ind] = (ab[1, ~ind] + 1) // -2

    return xx, yy


def get_elements_from_kbody_term(kbody_term: str) -> List[str]:
    """
    Return the atoms in the given k-body term.

    Parameters
    ----------
    kbody_term : str
        A str as the k-body term.

    Returns
    -------
    elements : List
        A list of str as the elements of the k-body term.

    """
    sel = [0]
    for i in range(len(kbody_term)):
        if kbody_term[i].isupper():
            sel.append(i + 1)
        else:
            sel[-1] += 1
    atoms = []
    for i in range(len(sel) - 1):
        atoms.append(kbody_term[sel[i]: sel[i + 1]])
    return atoms


def get_kbody_terms(elements: List[str],
                    angular=False,
                    symmetric=True):
    """
    Return ordered k-body terms (k=2 or k=2,3 if angular is True).

    Parameters
    ----------
    elements : List[str]
        A list of str as the target elements.
    angular : bool
        If True, 3-body terms will also be included.
    symmetric : SymmetricMode
        A boolean flag. If False, both ABA and AAB shall be included.

    Returns
    -------
    all_kbody_terms : List[str]
        A list of str as the ordered k-body terms.
    kbody_terms_for_element : Dict[str, List[str]]
        A dict of (element, List[str]) as the k-body terms for each element.
    elements : List[str]
        The sorted elements.

    """
    elements = sorted(list(set(elements)))
    n = len(elements)
    kbody_terms_for_element = {}
    for i in range(n):
        kbody_term = "{}{}".format(elements[i], elements[i])
        kbody_terms_for_element[elements[i]] = [kbody_term]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            kbody_term = "{}{}".format(elements[i], elements[j])
            kbody_terms_for_element[elements[i]].append(kbody_term)
    if angular:
        for i in range(n):
            center = elements[i]
            for j in range(n):
                if symmetric:
                    for k in range(j, n):
                        suffix = "".join(sorted([elements[j], elements[k]]))
                        kbody_term = f"{center}{suffix}"
                        kbody_terms_for_element[elements[i]].append(kbody_term)
                else:
                    for k in range(n):
                        kbody_term = f"{center}{elements[j]}{elements[k]}"
                        kbody_terms_for_element[elements[i]].append(kbody_term)
    all_kbody_terms = [
        x for x in chain(*[kbody_terms_for_element[element]
                           for element in elements])]
    return all_kbody_terms, kbody_terms_for_element, elements


def set_logging_configs(logfile="logfile", level=logging.INFO):
    """
    Setup the logging module.
    """
    LOGGING_CONFIG = {
        "version": 1,
        "formatters": {
            'file': {
                'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
            },
        },
        "handlers": {
            'file': {
                'class': 'logging.FileHandler',
                'level': level,
                'formatter': 'file',
                'filename': logfile,
                'mode': 'a',
            },
        },
        "root": {
            'handlers': ['file'],
            'level': level,
        },
        "disable_existing_loggers": False
    }
    dictConfig(LOGGING_CONFIG)


class ModeKeys:
    """Standard names for Estimator model modes.

    The following standard keys are defined:
    * `TRAIN`: training/fitting mode.
    * `EVAL`: testing/evaluation mode.
    * `PREDICT`: predication/inference mode.
    * `LAMMPS`: prediciton/inference mode for LAMMPS
    * `KMC`: prediciton/inference mode for TensorKMC
  """

    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'
    LAMMPS = 'lammps'
    KMC = 'kmc'
    PRECOMPUTE = 'precompute'

    @staticmethod
    def for_prediction(modekey):
        return modekey in ('infer', 'lammps', 'kmc', 'precompute')


class GraphKeys:
    """
    Standard names for variable collections.
    """

    # Variable keys
    DESCRIPTOR_VARIABLES = 'descriptor_variables'
    ATOMIC_NN_VARIABLES = 'atomic_nn_variables'
    ATOMIC_RES_NN_VARIABLES = 'atomic_res_nn_variables'
    EAM_ALLOY_NN_VARIABLES = 'eam_alloy_nn_variables'
    EAM_FS_NN_VARIABLES = 'eam_fs_nn_variables'
    EAM_POTENTIAL_VARIABLES = 'eam_potential_variables'
    DEEPMD_VARIABLES = "deepmd_variables"
    TERSOFF_VARIABLES = "tersoff_variables"

    # Metrics Keys
    TRAIN_METRICS = 'train_metrics'
    EVAL_METRICS = 'eval_metrics'


class AttributeDict(dict):
    """
    A subclass of `dict` with attribute-style access.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


RANDOM_STATE = 611


class Defaults:
    """
    A dataclass storing default parameters.
    """
    rc = 6.0
    k_max = 2

    eta = np.array([0.05, 4.0, 20.0, 80.0])
    omega = np.array([0.0])
    beta = np.array([0.005, ])
    gamma = np.array([1.0, -1.0])
    zeta = np.array([1.0, 4.0])

    cutoff_function = 'cosine'

    n_etas = 4
    n_betas = 1
    n_gammas = 2
    n_zetas = 2
    n_omegas = 1

    seed = RANDOM_STATE

    variable_moving_average_decay = 0.999

    activation = 'softplus'
    hidden_sizes = [64, 32]
    learning_rate = 0.01


def safe_select(a, b):
    """
    A helper function to return `a` if it is neither None nor empty.
    """
    if a is None:
        return b
    elif hasattr(a, '__len__'):
        if len(a) == 0:
            return b
    elif isinstance(a, str):
        if a == '':
            return b
    return a


def check_path(path):
    """
    Make sure the given path is accessible.
    """
    dst = dirname(path)
    if not isdir(dst):
        makedirs(dst)
    return path


def nested_get(d: dict, nested_keys: Union[str, List[str]]) -> Any:
    """
    Get the value from the dict `d` with a keypath (e.g `a.b.c`) or a list of
    nested keys (e.g ['a', 'b', 'c']).

    Parameters
    ----------
    d : dict
        A dict.
    nested_keys : str or List[str]
        A str as the key path or a list of str.

    Returns
    -------
    val : Any
        The value corresponding to the keypath.

    """
    if isinstance(nested_keys, str):
        nested_keys = nested_keys.split('.')
    obj = d
    for i, key in enumerate(nested_keys):
        if not hasattr(obj, "__getitem__"):
            return None
        obj = obj.get(key, None)
        if obj is None:
            return None
    return obj


def nested_set(d: dict, nested_keys: Union[str, List[str]], new_val):
    """
    Set the value of dict `d` with the given keypath or nested keys.
    """
    if isinstance(nested_keys, str):
        nested_keys = nested_keys.split('.')
    if not isinstance(d, dict):
        raise ValueError("`d` must be a dict")
    obj = d
    n = len(nested_keys)
    for i, key in enumerate(nested_keys):
        if i == n - 1:
            obj[key] = new_val
        else:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
