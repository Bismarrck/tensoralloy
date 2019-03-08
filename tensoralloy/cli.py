# coding=utf-8
"""
This module defines the command-line main function of `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import argparse
import glob
import warnings
import re
import os

from os.path import exists, dirname, join, basename, splitext

from tensoralloy.io.read import read_file
from tensoralloy.io.input import InputReader
from tensoralloy.test_utils import datasets_dir
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


class ExportModelProgram(CLIProgram):
    """
    The program for exporting models from checkpoint files.
    """

    @property
    def name(self):
        return "export"

    @property
    def help(self):
        return "Export a tensorflow checkpoint to model file(s)."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'ckpt',
            type=str,
            help="The checkpoint file or 'latest'"
        )
        subparser.add_argument(
            '-i', '--input',
            type=str,
            default=None,
            help="The corresponding input toml file."
        )

        super(ExportModelProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        def func(args: argparse.Namespace):
            if args.ckpt.lower() == 'latest':
                ckpt = tf.train.latest_checkpoint(".")
            else:
                ckpt = args.ckpt

            if not tf.train.checkpoint_exists(ckpt):
                raise IOError(f"The checkpoint file {ckpt} cannot be accessed!")

            if args.input is None:
                model_dir = dirname(ckpt)
                candidates = list(glob.glob(join(model_dir, "*.toml")))
                if len(candidates) > 1:
                    warnings.warn(f"More than one TOML file in {model_dir}. "
                                  f"Only {candidates[0]} will be used.")
                filename = candidates[0]
            else:
                filename = args.input
                model_dir = dirname(filename)

            step_tag = splitext(basename(ckpt).split('-')[-1])[0]

            configs = InputReader(filename)
            database_file = configs['dataset.sqlite3']
            database_name = basename(database_file)

            if not exists(database_file):
                if exists(join(model_dir, database_name)):
                    database_file = join(model_dir, database_name)
                elif exists(join(datasets_dir(), database_name)):
                    database_file = join(datasets_dir(), database_name)
                else:
                    raise IOError(f"The Sqlite3 database {database_file} "
                                  f"cannot be accessed!")

            configs['train.model_dir'] = model_dir
            configs['dataset.sqlite3'] = database_file

            manager = TrainingManager(configs, validate_tfrecords=False)
            manager.export(ckpt, tag=step_tag)
        return func


class StopExperimentProgram(CLIProgram):
    """
    Stop an experiment by directly terminating the corresponding process.
    """

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "stop"

    @property
    def help(self):
        """
        The help message.
        """
        return "Terminate a running experiment."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            "model_dir",
            type=str,
            help="The model dir of the experiment."
        )

        super(StopExperimentProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        def func(args: argparse.Namespace):
            logfile = join(args.model_dir, "logfile")
            if not exists(logfile):
                raise IOError(f"{logfile} cannot be accessed")
            pid_patt = re.compile(r".*tensorflow\s+INFO\s+pid=(\d+)")
            with open(logfile) as fp:
                for number, line in enumerate(fp):
                    m = pid_patt.search(line)
                    if m:
                        pid = m.group(1)
                        break
                    if number == 10:
                        raise IOError(f"{logfile} maybe corrupted!")
            code = os.system(f'kill -9 {pid}')
            if code == 0:
                print(f"pid={pid} killed. "
                      f"The experiment in {args.model_dir} terminated.")
            else:
                print(f"Failed to stop {args.model_dir}: error_code = {code}")
        return func


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
                 RunExperimentProgram(),
                 ExportModelProgram(),
                 StopExperimentProgram()):
        subparser = subparsers.add_parser(prog.name, help=prog.help)
        prog.config_subparser(subparser)

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
    else:
        args.func(args)
