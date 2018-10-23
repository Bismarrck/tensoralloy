# coding=utf-8
"""
This module defines miscellaneous functions.
"""
from __future__ import print_function, absolute_import

from unittest import SkipTest
from os.path import dirname, isdir
from os import makedirs

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The random seed.
RANDOM_STATE = 611


def skip(func):
    """
    A decorator for skipping tests.
    """
    def _():
        raise SkipTest("Test %s is skipped" % func.__name__)
    _.__name__ = func.__name__
    return _


def check_path(path):
    """ Make sure the given path is accessible. """
    dst = dirname(path)
    if not isdir(dst):
        makedirs(dst)
    return path

