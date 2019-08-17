# coding=utf-8
"""
This module defines unit tests related functions and vars.
"""
from __future__ import print_function, absolute_import

from os.path import join, dirname, abspath

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def datasets_dir(absolute=False):
    """
    Return the directory of `datasets`. Built-in datasets can be found here.
    """
    path = join(dirname(__file__), "..", "datasets")
    if absolute:
        path = abspath(path)
    return path


def project_dir(absolute=False):
    """
    Return the root directory of this project.
    """
    path = join(dirname(__file__), "..")
    if absolute:
        path = abspath(path)
    return path
