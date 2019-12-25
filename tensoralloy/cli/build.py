#!coding=utf-8
"""
Command-line programs under the `build` scope.
"""
from __future__ import print_function, absolute_import

import numpy as np
import argparse
import json

from tensoralloy.cli.cli import CLIProgram
from tensoralloy.io.read import read_file
from tensoralloy.io import lammps

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
            exclude_types = args.exclude_types
            exclude_pairs = args.exclude_pairs
            filename = args.filename

            if args.output is None:
                output = f"{filename}.json"
            else:
                output = args.output

            if args.format == "meam/spline":
                if args.old:
                    if args.element is None:
                        raise ValueError(
                            "The arg 'element' must be set if '--old' is true.")
                    lmp = lammps.read_old_meam_spline_file(
                        filename, args.element)
                else:
                    lmp = lammps.read_meam_spline_file(filename)
            else:
                if args.format == "adp":
                    lmp = lammps.read_adp_setfl(filename)
                else:
                    lmp = lammps.read_eam_alloy_setfl(filename)

            data = {}

            def update_field(key, pot):
                """
                A helper function to update the dict `data`.
                """
                size = len(getattr(lmp, pot)[key].x)
                if args.npoints is not None:
                    if args.npoints < size:
                        idx = np.round(
                            np.linspace(0, size - 1, args.npoints)).astype(int)
                    else:
                        idx = np.arange(0, size).astype(int)
                    data[f"{key}.{pot}.x"] = \
                        getattr(lmp, pot)[key].x[idx].tolist()
                    data[f"{key}.{pot}.y"] = \
                        getattr(lmp, pot)[key].y[idx].tolist()
                else:
                    if args.interval > size:
                        step = 1
                    else:
                        step = args.interval
                    data[f"{key}.{pot}.x"] = \
                        getattr(lmp, pot)[key].x[::step].tolist()
                    data[f"{key}.{pot}.y"] = \
                        getattr(lmp, pot)[key].y[::step].tolist()
                data[f"{key}.{pot}.bc_start"] = getattr(lmp, pot)[key].bc_start
                data[f"{key}.{pot}.bc_end"] = getattr(lmp, pot)[key].bc_end

            if exclude_types:
                exclude_types = [x.strip().lower()
                                 for x in exclude_types.split(",")]
            else:
                exclude_types = []
            if exclude_pairs:
                exclude_pairs = [x.strip().lower()
                                 for x in exclude_pairs.split(",")]
            else:
                exclude_pairs = []

            n = len(getattr(lmp, "elements"))
            for i in range(n):
                a = getattr(lmp, "elements")[i]
                if a not in exclude_pairs:
                    for ptype in ("rho", "embed"):
                        if ptype not in exclude_types:
                            update_field(a, ptype)
                    if args.format == "meam/spline":
                        if "fs" not in exclude_types:
                            update_field(a, "fs")
                for j in range(i, n):
                    b = getattr(lmp, "elements")[j]
                    ab = "".join(sorted((a, b)))
                    if ab not in exclude_pairs:
                        if "phi" not in exclude_types:
                            update_field(ab, "phi")
                        if args.format == "adp":
                            for ptype in ("dipole", "quadrupole"):
                                if ptype not in exclude_types:
                                    update_field(ab, ptype)
                        elif args.format == "meam/spline":
                            if "gs" not in exclude_types:
                                update_field(ab, "gs")
            with open(output, "w") as fp:
                json.dump(data, fp, indent=2)

        return func

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            "filename",
            type=str,
            help="A lammps potential file."
        )
        subparser.add_argument(
            "--output",
            type=str,
            default=None,
            help="The output initial guess csv file."
        )
        subparser.add_argument(
            "--format",
            type=str,
            choices=['eam', 'adp', 'meam/spline'],
            help="The format of the lammps potential file."
        )
        group = subparser.add_mutually_exclusive_group()
        group.add_argument(
            "--interval",
            type=int,
            default=None,
            help="The slice interval for selecting reference data points."
        )
        group.add_argument(
            "--npoints",
            type=int,
            default=25,
            help="The number of data points for building spline functions. "
                 "Has the higher priority."
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

        group = subparser.add_argument_group("meam/spline")
        group.add_argument(
            "--element",
            default=None,
            type=str,
            help="Required if `--old` is added."
        )
        group.add_argument(
            "--old",
            action="store_true",
            default=False,
            help="Add this flag if the meam/spline file is the old "
                 "single-specie format"
        )

        super(BuildSplineGuessProgram, self).config_subparser(subparser)
