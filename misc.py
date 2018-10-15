# coding=utf-8
"""
This module defines miscellaneous functions.
"""
from __future__ import print_function, absolute_import

from unittest import SkipTest

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def skip(func):
    """
    A decorator for skipping tests.
    """
    def _():
        raise SkipTest("Test %s is skipped" % func.__name__)
    _.__name__ = func.__name__
    return _
