#!coding=utf-8
"""
TensorAlloy-DeePMD utility programs.
"""
from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np
import argparse

from collections import Counter
from itertools import chain, repeat
from os import makedirs
from os.path import exists, join, splitext
from matplotlib import pyplot as plt

from tensoralloy.cli.cli import CLIProgram
from tensoralloy.io import read_file

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class DeepmdKitProgram(CLIProgram):
    """
    A collection of programs for interacting with Deepmd-Kit.
    """

    def __init__(self):
        """
        Initiazliation method.
        """
        super(DeepmdKitProgram, self).__init__()

        self._programs = [
            BuildDeepmdKitDataProgram(),
            PlotDeepmdKitLearningCurveProgram(),
        ]

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "deepmd"

    @property
    def help(self):
        """
        The help message
        """
        return "TensorAlloy-DeePMD utility programs."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the subparaser.
        """
        subsubparsers = subparser.add_subparsers(title=self.name,
                                                 help=self.help)
        for prog in self._programs:
            subsubparser = subsubparsers.add_parser(prog.name, help=prog.help)
            prog.config_subparser(subsubparser)
        super(DeepmdKitProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """ The main function. This is just a program-collection, so this
        function is empty. """
        def func(_):
            pass
        return func


class BuildDeepmdKitDataProgram(CLIProgram):
    """
    Build deepmd-kit data files from an extxyz file or a sqlite3 database.
    """

    @property
    def name(self):
        return "build"

    @property
    def help(self):
        return "Build deepmd-kit data files from an extxyz file or a sqlite3 " \
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
            nnl_max = db.get_nnl_max(rc=args.rc, allow_calculation=True)
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
                fp.write(f"sel@{args.rc:.2f}: {[nnl_max] * len(elements)}\n")
            print(f"Type map: {str(elements)}")
            print(f"Sel [{args.rc:.2f}]: {str([nnl_max] * len(elements))}")
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
        subparser.add_argument(
            '--rc',
            default=6.0,
            type=float,
            help="The cutoff radius for detecting \"sel\""
        )
        super(BuildDeepmdKitDataProgram, self).config_subparser(subparser)


class PlotDeepmdKitLearningCurveProgram(CLIProgram):
    """
    The program for plotting deepmd-kit learning curves.
    """

    @property
    def name(self):
        """
        The name of this program.
        """
        return "plot"

    @property
    def help(self):
        """
        The help message.
        """
        return "Plot a deepmd-kit learning curve."

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            logfile = args.logfile
            labels = None
            nlabels = 0
            data = {}
            hlabels = {
                "l2_tst": "Loss (test)",
                "l2_trn": "Loss (train)",
                "l2_e_tst": "Energy (test)",
                "l2_e_trn": "Energy (train)",
                "l2_f_tst": "Force (test)",
                "l2_f_trn": "Force (train)",
                "l2_v_tst": "Virial (test)",
                "l2_v_trn": "Virial (train)",
                "lr": "Learning rate"
            }
            subplots_kwargs = {
                2: {"nrows": 2, "ncols": 1, "figsize": [5, 8]},
                3: {"nrows": 2, "ncols": 2, "figsize": [10, 8]},
                4: {"nrows": 2, "ncols": 2, "figsize": [10, 8]},
                5: {"nrows": 2, "ncols": 3, "figsize": [13, 4]},
                6: {"nrows": 2, "ncols": 3, "figsize": [13, 4]},
                7: {"nrows": 3, "ncols": 3, "figsize": [12, 10]},
                8: {"nrows": 3, "ncols": 3, "figsize": [12, 10]},
                9: {"nrows": 3, "ncols": 3, "figsize": [12, 10]},
            }
            with open(logfile) as fp:
                for line in fp:
                    line = line.strip()
                    if line.startswith("#"):
                        labels = line.split()[1:]
                        nlabels = len(labels)
                        data = {label: [] for label in labels}
                        continue
                    splits = line.split()
                    if len(splits) == nlabels:
                        for i, label in enumerate(labels):
                            ltype = float if i else int
                            data[label].append(ltype(splits[i]))
            df = pd.DataFrame(data)
            if args.indices is not None:
                indices = [int(x) for x in args.indices.split(",")]
                cols = [labels[i] for i in indices if i > 0]
            elif args.cols is not None:
                if args.cols == "all":
                    cols = labels[1:]
                else:
                    cols = args.cols.split(",")
                for col in cols:
                    assert col in labels and col != "batch"
            else:
                cols = ["l2_f_tst"]
            ncols = len(cols)

            def plot_col_on_ax(col_ax: plt.Axes, col_idx: int):
                """
                Plot the column on the given ax.
                """
                x = df["batch"]
                y = df[cols[col_idx]]
                col_ax.plot(x, y, label=cols[col_idx])
                col_ax.set_xlabel(f"Batch", fontsize=12)
                col_ax.set_ylabel(f"{hlabels[cols[col_idx]]}", fontsize=12)
                col_ax.set_xscale('log')
                col_ax.set_yscale('log')
                col_ax.legend()

            if ncols == 1 or args.independent:
                for i in range(ncols):
                    fig, ax = plt.subplots(1, 1, figsize=[5, 4])
                    plot_col_on_ax(ax, i)
                    plt.tight_layout()
                    plt.savefig(f"{cols[i]}.png", dpi=300)
                    if args.show:
                        plt.show()
                    else:
                        plt.close(fig)
            else:
                fig, axes = plt.subplots(**subplots_kwargs[ncols],
                                         squeeze=False)
                for i, ax in enumerate(axes.flatten()):
                    if i < ncols:
                        plot_col_on_ax(ax, i)
                    else:
                        ax.set_visible(False)
                plt.tight_layout()
                plt.savefig(f"{splitext(args.logfile)[0]}.png", dpi=300)
                if args.show:
                    plt.show()
                else:
                    plt.close(fig)
        return func

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'logfile',
            type=str,
            help="Specify the learning curve file to read.",
        )
        subparser.add_argument(
            "--cols",
            type=str,
            default=None,
            help="Comma-separated column names or 'all' for all columns."
        )
        subparser.add_argument(
            "--indices",
            type=str,
            default=None,
            help="Comma-separated int string indicating the columns to plot."
        )
        subparser.add_argument(
            "--independent",
            action="store_true",
            default=False,
            help="Plot each curve in an independent plot if this flag is set."
        )
        subparser.add_argument(
            "--show",
            action="store_true",
            default=False,
            help="Show the plots directly."
        )
        super(PlotDeepmdKitLearningCurveProgram, self).config_subparser(
            subparser)
