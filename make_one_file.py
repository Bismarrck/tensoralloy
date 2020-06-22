#!coding=utf-8
"""
A script to merge all Python files and TOML files into a single txt file.
"""
from __future__ import print_function, absolute_import

from os import walk
from os.path import join, splitext

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

out = ""

for root, sub_folders, files in walk("."):
    if root.startswith("./"):
        root = root[2:]
    tops = root.split("/")
    if tops[0] in (".vscode", "experiments", "models", ".git", ".idea",
                   "test_files"):
        continue
    if tops[-1] == "__pycache__":
        continue
    for afile in files:
        if afile == "test.py" or afile == "make_one_file.py":
            continue
        if afile.startswith("."):
            continue
        ext = splitext(afile)[1]
        if ext in (".py", ".toml", ".md", ".txt"):
            filename = join(root, afile)
            with open(filename) as fp:
                if root == ".":
                    header = f"tensoralloy/{afile}"
                else:
                    header = f"tensoralloy/{root}/{afile}"
                out += f"Path={header}\n{fp.read()}\n"

with open("all_codes", "w") as fp:
    fp.write(out)
