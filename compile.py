# coding=utf-8
"""
Compile the entire package to a cython static library.
"""
from __future__ import print_function, absolute_import

import os
import glob
from os.path import join, isdir, splitext

from distutils.core import setup
from distutils.cmd import Command
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils.build_ext import new_build_ext as build_ext

from setup import __version__

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# `cython (<=0.29.1)` can not handle `dataclass` or `nametuple` correctly.
avoid_to_cythonize = [
    'tensoralloy/transformer/indexed_slices.py',
    'tensoralloy/descriptor/cutoff.py',
    'tensoralloy/cli.py',
]
ext_modules = []
files_to_delete = []
packages = []


for package in find_packages(exclude=['datasets', 'test_files', 'tools']):
    folders = package.split('.')
    if folders[-1] == 'tests':
        continue
    packages.append(package)
    cwd = join(*folders)
    for name in os.listdir(cwd):
        path = join(cwd, name)
        if path in avoid_to_cythonize:
            continue
        if name.startswith('.') or \
                isdir(path) or \
                name.endswith('.toml') or \
                name.endswith(".so") or \
                name.endswith('.c'):
            continue
        ext_modules.append(Extension(f'{package}.{splitext(name)[0]}', [path]))

        files_to_delete.append(f"{splitext(path)[0]}.cpython-*.so")
        files_to_delete.append(f"{splitext(path)[0]}.c")


class CleanExt(Command):
    """
    Delete the generated .c and .so files.
    """

    description = "Delete the generated .c and .so files."
    user_options = []

    def initialize_options(self):
        """
        Set default values for all the options that this command supports.
        """
        pass

    def finalize_options(self):
        """
        Set final values for all the options that this command supports.
        """
        pass

    def run(self):
        """
        Run this command.
        """
        for pattern in files_to_delete:
            paths = list(glob.glob(pattern))
            if len(paths) == 1:
                os.remove(paths[0])
                print(f"Delete: {paths[0]}")
            elif len(paths) > 1:
                raise RuntimeError(
                    f"File pattern {pattern} returns {len(paths)} files.")


setup(
    name='tensoralloy',
    cmdclass={
        'clean_ext': CleanExt,
        'build_ext': build_ext
    },
    ext_modules=ext_modules,
    author=__author__,
    author_email=__email__,
    version=__version__,
    description="Tensor-graph based machine learning framework for alloys.",
    packages=packages,
    entry_points={'console_scripts': ['tensoralloy=tensoralloy.cli:main']},
    include_package_data=False,
    python_requires=">=3.6.5",
)
