#!coding=utf-8
"""
Command-line programs under the `build` scope.
"""
from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np
import argparse

from tensoralloy.cli.cli import CLIProgram
from tensoralloy.io.read import read_file
from tensoralloy.io.lammps import read_eam_alloy_setfl, read_adp_setfl

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
            BuildSplineGuessProgram(),
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


class BuildSplineGuessProgram(CLIProgram):
    """
    Build the initial guess file of cubic spline potentials.
    """

    @property
    def name(self):
        return "spline"

    @property
    def help(self):
        return "Build initial guess for cubic spline potentials"

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            is_adp = False
            try:
                setfl = read_adp_setfl(args.setfl)
            except Exception:
                try:
                    setfl = read_eam_alloy_setfl(args.setfl)
                except Exception as excp:
                    raise excp
            else:
                is_adp = True
            data = {}

            def update_field(key, pot):
                """
                A helper function to update the dict `data`.
                """
                data[f"{key}.{pot}.x"] = getattr(setfl, pot)[key][0][::step]
                data[f"{key}.{pot}.y"] = getattr(setfl, pot)[key][1][::step]

            if args.exclude_types:
                exclude_types = [x.strip().lower()
                                 for x in args.exclude_types.split(",")]
            else:
                exclude_types = []
            if args.exclude_pairs:
                exclude_pairs = [x.strip().lower()
                                 for x in args.exclude_pairs.split(",")]
            else:
                exclude_pairs = []
            step = args.interval
            n = len(setfl.elements)
            for i in range(n):
                a = setfl.elements[i]
                if a not in exclude_pairs:
                    for ptype in ("rho", "embed"):
                        if ptype not in exclude_types:
                            update_field(a, ptype)
                for j in range(i, n):
                    b = setfl.elements[j]
                    ab = "".join(sorted((a, b)))
                    if ab not in exclude_pairs:
                        if "phi" not in exclude_types:
                            update_field(ab, "phi")
                        if is_adp:
                            for ptype in ("dipole", "quadrupole"):
                                if ptype not in exclude_types:
                                    update_field(ab, ptype)
            if len(set([len(y) for y in data.values()])) == 1:
                if not args.output:
                    output = f"{args.setfl}.guess"
                else:
                    output = args.output
                df = pd.DataFrame(data)
                df.to_csv(output, index=None)
            else:
                output = f"{args.setfl}.npz"
                print(f"Save the guess to {output} because not all arrays have "
                      f"the same length")
                np.savez(output, **data)
        return func

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            "setfl",
            type=str,
            help="A lammps setfl potential file."
        )
        subparser.add_argument(
            "-o", "--output",
            type=str,
            default=None,
            help="The output initial guess csv file."
        )
        subparser.add_argument(
            "--interval",
            type=int,
            default=100,
            help="The slice interval for selecting reference data points."
        )
        subparser.add_argument(
            "--exclude-types",
            default=None,
            type=str,
            help="A comma-separated string as the excluded "
                 "potential types (e.g. 'rho,embed')"
        )
        subparser.add_argument(
            "--exclude-pairs",
            default=None,
            type=str,
            help="A comma-separated string as the excluded "
                 "elements or interaction types (e.g. 'Ni,NiMo')"
        )
        super(BuildSplineGuessProgram, self).config_subparser(subparser)
