# coding=utf-8
"""
The setup module.
"""
from __future__ import print_function, absolute_import

from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


__version__ = "1.0"
__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming,PyAttributeOutsideInit
class bdist_wheel(_bdist_wheel):

    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        # Mark us as not a pure python package
        self.root_is_pure = False


if __name__ == "__main__":

    packages = find_packages()
    if not packages:
        packages = ['tensoralloy']

    setup(
        name="tensoralloy",
        author=__author__,
        author_email=__email__,
        version=__version__,
        description="Tensor-graph based machine learning framework for alloys.",
        packages=packages,
        include_package_data=True,
        entry_points={
            'console_scripts': ['tensoralloy=tensoralloy.cli:main']
        },
        python_requires=">=3.7.0",
    )
