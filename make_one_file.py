#!coding=utf-8
"""
A script to merge all Python files and TOML files into a single txt file.
"""
from __future__ import print_function, absolute_import

from os import walk
from os.path import join

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

content = ""

for root, sub_folders, files in walk("."):
    if root in ("datasets", "examples", "experiments", "models"):
        continue
    for afile in files:
        if afile == "test.py" or afile == "make_one_file.py":
            continue
        if afile.endswith("py") or afile.endswith("toml"):
            filename = join(root, afile)
            with open(filename) as fp:
                content += f"tensoralloy/{afile}\n" + fp.read() + "\n"

with open("all_codes", "w") as fp:
    fp.write(content)
