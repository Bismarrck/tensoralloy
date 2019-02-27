# coding=utf-8
"""
This module defines the command-line main function of `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import argparse

from tensoralloy.io.read import read_file
from tensoralloy.train import TrainingManager


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


class RunExperimentProgram(CLIProgram):
    """
    The program for running training experiments.
    """

    @property
    def name(self):
        return "run"

    @property
    def help(self):
        return "Run an experiment"

    @property
    def main_func(self):
        """
        Return the main function for running experiments.
        """
        def func(args: argparse.Namespace):
            manager = TrainingManager(
                args.filename)
            manager.train_and_evaluate(debug=args.debug)
            manager.export()
        return func

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'filename',
            type=str,
            help="A cfg file to read."
        )
        subparser.add_argument(
            "--debug",
            action='store_true',
            default=False,
            help="Enabled the debugging mode.",
        )
        super(RunExperimentProgram, self).config_subparser(subparser)


class BuildDatabaseProgram(CLIProgram):
    """
    The program for building databases from files.
    """

    @property
    def name(self):
        return "build"

    @property
    def help(self):
        return "Build a sqlite3 database from a extxyz file"

    @property
    def main_func(self):
        """
        Return the main function for building databases.
        """
        def func(args: argparse.Namespace):
            read_file(args.filename,
                      units={'energy': args.energy_unit,
                             'forces': args.forces_unit,
                             'stress': args.stress_unit},
                      num_examples=args.num_examples,
                      verbose=True)
        return func

    def config_subparser(self, subparser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'filename',
            type=str,
            help="Specify the xyz or extxyz file to read.",
        )
        subparser.add_argument(
            '--num-examples',
            default=None,
            type=int,
            help="Set the maximum number of examples to read."
        )
        subparser.add_argument(
            '--energy-unit',
            type=str,
            default='eV',
            choices=('eV', 'Hartree', 'kcal/mol'),
            help='The unit of the energies in the file'
        )
        subparser.add_argument(
            '--forces-unit',
            type=str,
            default='eV/Angstrom',
            choices=['kcal/mol/Angstrom', 'kcal/mol/Bohr', 'eV/Bohr',
                     'eV/Angstrom', 'Hartree/Bohr', 'Hartree/Angstrom'],
            help='The unit of the atomic forces in the file.'
        )
        subparser.add_argument(
            '--stress-unit',
            type=str,
            default='eV/Angstrom**3',
            choices=['GPa', 'kbar', 'eV/Angstrom**3'],
            help='The unit of the stress tensors in the file.',
        )
        super(BuildDatabaseProgram, self).config_subparser(subparser)


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar='Build a database or run an experiment.'
    )

    for prog in (BuildDatabaseProgram(),
                 RunExperimentProgram()):
        subparser = subparsers.add_parser(prog.name, help=prog.help)
        prog.config_subparser(subparser)

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
    else:
        args.func(args)
