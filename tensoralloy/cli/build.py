#!coding=utf-8
"""
Command-line programs under the `build` scope.
"""
from __future__ import print_function, absolute_import

import argparse

from tensoralloy.cli.cli import CLIProgram
from tensoralloy.io.read import read_file

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class BuildProgram(CLIProgram):
    """
    A collection of programs under the scope `build`.
    """

    def __init__(self):
        """
        Initialization method.
        """
        self._programs = [
            BuildDatabaseProgram(),
        ]

    @property
    def name(self):
        """
        The name of this program.
        """
        return "build"

    @property
    def help(self):
        """
        The help message.
        """
        return "Build databases or files"

    @property
    def main_func(self):
        """
        The main function. This is just a program container, so this function
        is empty. """
        def func(_):
            pass
        return func

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subsubparsers = subparser.add_subparsers(title=self.name,
                                                 help=self.help)
        for prog in self._programs:
            subsubparser = subsubparsers.add_parser(prog.name, help=prog.help)
            prog.config_subparser(subsubparser)
        super(BuildProgram, self).config_subparser(subparser)


class BuildDatabaseProgram(CLIProgram):
    """
    The program for building databases from files.
    """

    @property
    def name(self):
        return "database"

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
                      file_type=args.format,
                      units={'energy': args.energy_unit,
                             'forces': args.forces_unit,
                             'stress': args.stress_unit},
                      num_examples=args.num_examples,
                      append_to=args.append_to,
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
            '--format',
            type=str,
            default=None,
            choices=['db', 'polar', 'extxyz', 'xyz', 'stepmax'],
            help="The file format"
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
        subparser.add_argument(
            "--append-to",
            type=str,
            default=None,
            help="The database file to append to."
        )
        subparser.add_argument(
            "--fmax",
            type=float,
            default=None,
            help="The maximum force threshold."
        )
        super(BuildDatabaseProgram, self).config_subparser(subparser)
