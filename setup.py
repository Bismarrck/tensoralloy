# coding=utf-8
"""
The setup module.
"""
from __future__ import print_function, absolute_import

import sys
from setuptools import setup, find_packages

__version__ = "2018.11.26"
__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


if sys.version_info < (3, 7):
    sys.exit('Python < 3.7 is not supported')


setup(
    name="tensoralloy",
    author=__author__,
    author_email=__email__,
    version=__version__,
    description="Tensor-graph based machine learning framework for alloys.",
    packages=find_packages(exclude=['examples', 'doc']),
    include_package_data=True,
)
