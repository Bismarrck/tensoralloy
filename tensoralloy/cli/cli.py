# coding=utf-8
"""
The base `CLIProgram` class.
"""
from __future__ import print_function, absolute_import

import argparse

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class CLIProgram:
    """
    This class represents a command line program.
    """

    @property
    def main_func(self):
        """
        An abstract property. Return the main function for this program.
        """
        raise NotImplementedError(
            "This property should be implemented by a subclass")

    @property
    def name(self):
        """
        Return the name of this program.
        """
        return "cli"

    @property
    def help(self):
        """
        Return the help message for this program.
        """
        return ""

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.set_defaults(func=self.main_func)
