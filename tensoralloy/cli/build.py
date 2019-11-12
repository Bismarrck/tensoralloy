#!coding=utf-8
"""
Command-line programs under the `build` scope.
"""
from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np
import argparse

from os.path import exists, join
from os import makedirs
from collections import Counter
from itertools import repeat, chain

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
            BuildDeepkitDataProgram()
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


class BuildDeepkitDataProgram(CLIProgram):
    """
    Convert an extxyz file or a sqlite3 database to DeepKit data files.
    """

    @property
    def name(self):
        return "deepkit"

    @property
    def help(self):
        return "Build Deepkit data files from an extxyz file or a sqlite3 " \
               "database."

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            if not exists(args.outdir):
                makedirs(args.outdir)
            db = read_file(args.target)
            print(f"Source: {args.target}")
            elements = sorted(db.max_occurs.keys())
            size = len(db)
            table = {}
            name = {}

            # Check the total number of systems
            for atoms_id in range(1, 1 + size):
                atoms = db.get_atoms(id=atoms_id)
                symbols = atoms.get_chemical_symbols()
                c = Counter(symbols)
                system = " ".join(
                    map(str,
                        chain(*[repeat(elements.index(e), c[e]) for e in c])))
                table[system] = table.get(system, []) + [atoms_id]
                name[system] = "".join([f"{e}{c[e]}" for e in elements])

            def floats2str(x, n1=12, n2=6):
                fmt = "{:%d.%df}" % (n1, n2)
                return " ".join([fmt.format(xi)
                                 for xi in np.asarray(x).flatten()]) + "\n"

            # Loop through each system
            for i, (system, id_list) in enumerate(table.items()):
                sys_dir = join(args.outdir, f"system.{i:03d}")
                if not exists(sys_dir):
                    makedirs(sys_dir)
                print(f"system.{i:03d}: {name[system]}, {len(id_list)}")
                h_fp = open(join(sys_dir, "box.raw"), "w")
                r_fp = open(join(sys_dir, "coord.raw"), "w")
                e_fp = open(join(sys_dir, "energy.raw"), "w")
                if db.has_forces:
                    f_fp = open(join(sys_dir, "force.raw"), "w")
                else:
                    f_fp = None
                if db.has_stress:
                    v_fp = open(join(sys_dir, "virial.raw"), "w")
                else:
                    v_fp = None
                for atoms_id in id_list:
                    atoms = db.get_atoms(id=atoms_id)
                    symbols = atoms.get_chemical_symbols()
                    h_fp.write(floats2str(atoms.cell))
                    e_fp.write(f"{atoms.get_potential_energy():.8g}\n")
                    if db.has_stress:
                        volume = atoms.get_volume()
                        virial = atoms.get_stress(voigt=False) * volume
                        v_fp.write(floats2str(virial, 12, 6))
                    order = np.argsort([elements.index(e) for e in symbols])
                    r_fp.write(floats2str(atoms.positions[order], 12, 6))
                    if db.has_forces:
                        f_fp.write(floats2str(atoms.get_forces()[order], 12, 6))
                t_fp = open(join(sys_dir, "type.raw"), "w")
                t_fp.write(f"{system}\n")
                t_fp.close()
                e_fp.close()
                h_fp.close()
                r_fp.close()
                if db.has_forces:
                    f_fp.close()
                if db.has_stress:
                    v_fp.close()
            with open(join(args.outdir, "metadata"), "w+") as fp:
                fp.write(f"type_map: {str(elements)}\n")
                fp.write(f"sel: {str([db.max_occurs[e] for e in elements])}\n")
            print(f"Type map: {str(elements)}")
            print(f"Max occurs: {str([db.max_occurs[e] for e in elements])}")
        return func

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'target',
            type=str,
            help="Specify the extxyz or database file to read.",
        )
        subparser.add_argument(
            '--outdir',
            default=".",
            type=str,
            help="Set the output dir."
        )
        super(BuildDeepkitDataProgram, self).config_subparser(subparser)


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
            df = pd.DataFrame(data)
            df.to_csv(args.output, index=None)
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
            "output",
            type=str,
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
