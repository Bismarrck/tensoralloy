# coding=utf-8
"""
The setup module.
"""
from __future__ import print_function, absolute_import

import sys
from setuptools import setup, find_packages

__version__ = "1.0.0"
__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


if sys.version_info < (3, 6, 5):
    sys.exit('Python < 3.6.5 is not supported')


if __name__ == "__main__":

    setup(
        name="tensoralloy",
        author=__author__,
        author_email=__email__,
        version=__version__,
        description="Tensor-graph based machine learning framework for alloys.",
        packages=find_packages(),
        include_package_data=False,
        python_requires=">=3.6.5",
    )
