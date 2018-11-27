# coding=utf-8
"""
This module defines utility functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import logging
from itertools import chain
from logging.config import dictConfig
from typing import List, Dict, Tuple

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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


def get_kbody_terms(elements: List[str], k_max=3) -> Tuple[List[str],
                                                           Dict[str, List[str]],
                                                           List[str]]:
    """
    Given a list of unique elements, construct all possible k-body terms and the
    dict mapping terms to each type of element.

    Parameters
    ----------
    elements : List[str]
        A list of str as the ordered unique elements.
    k_max : int
        The maximum k for the many-body expansion.

    Returns
    -------
    all_terms : List[str]
        A list of str as all k-body terms.
    kbody_terms : Dict[str, List[str]]
        A dict of (element, terms) mapping k-body terms to each type of element.
    elements : List[str]
        A list of str as the ordered unique elements.

    """
    elements = sorted(list(set(elements)))
    n = len(elements)
    k_max = max(k_max, 1)
    kbody_terms = {}
    for i in range(n):
        kbody_term = "{}{}".format(elements[i], elements[i])
        kbody_terms[elements[i]] = [kbody_term]
    if k_max >= 2:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                kbody_term = "{}{}".format(elements[i], elements[j])
                kbody_terms[elements[i]].append(kbody_term)
    if k_max >= 3:
        for i in range(n):
            center = elements[i]
            for j in range(n):
                for k in range(j, n):
                    suffix = "".join(sorted([elements[j], elements[k]]))
                    kbody_term = "{}{}".format(center, suffix)
                    kbody_terms[elements[i]].append(kbody_term)
    if k_max >= 4:
        raise ValueError("`k_max>=4` is not supported yet!")
    all_terms = [
        x for x in chain(*[kbody_terms[element] for element in elements])]
    return all_terms, kbody_terms, elements


def set_logging_configs(logfile="logfile"):
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
                'level': logging.INFO,
                'formatter': 'file',
                'filename': logfile,
                'mode': 'a',
            },
        },
        "root": {
            'handlers': ['file'],
            'level': logging.INFO,
        },
        "disable_existing_loggers": False
    }
    dictConfig(LOGGING_CONFIG)
