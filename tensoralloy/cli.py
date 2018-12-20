# coding=utf-8
"""
This module defines the command-line main function of `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import argparse

import tensoralloy.io.read
import tensoralloy.train.training


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar='Build a database or run an experiment.'
    )

    subparser = subparsers.add_parser(
        'build',
        help="Build a sqlite3 database from a extxyz file."
    )
    tensoralloy.io.read.config_parser(subparser)
    subparser.set_defaults(func=tensoralloy.io.read.main)

    subparser = subparsers.add_parser(
        'run',
        help="Run an experiment."
    )
    tensoralloy.train.training.config_parser(subparser)
    subparser.set_defaults(func=tensoralloy.train.training.main)

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
    else:
        args.func(args)
