#!coding=utf-8
"""
Command-line programs under the `export` scope.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import argparse
import glob
import warnings

from os.path import dirname, join, splitext, basename, exists

from tensoralloy import InputReader, TrainingManager
from tensoralloy.cli.cli import CLIProgram
from tensoralloy.test_utils import datasets_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class ExportModelProgram(CLIProgram):
    """
    The program for exporting models from checkpoint files.
    """

    @property
    def name(self):
        """
        The name of this program.
        """
        return "export"

    @property
    def help(self):
        """
        The help message of this program.
        """
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
        subparser.add_argument(
            '--no-ema',
            action='store_true',
            default=False,
            help="If this flag is given, EMA variables will be disabled."
        )
        subparser.add_argument(
            '--mode',
            default='infer',
            choices=['infer', 'lammps', 'kmc'],
            type=str,
            help="Export mode: infer, lammps or kmc"
        )

        group = subparser.add_argument_group("KMC")
        group.add_argument(
            "--nnl",
            type=int,
            help="The nnl_max parameter for TensorKMC")

        group = subparser.add_argument_group("EAM")
        group.add_argument(
            '--r0',
            type=float,
            default=0.0,
            help="The initial 'r' for plotting pair and density functions."
        )
        group.add_argument(
            '--rt',
            type=float,
            default=None,
            help="The final 'r' for plotting pair and density functions."
        )
        group.add_argument(
            '--rho0',
            type=float,
            default=0.0,
            help="The initial 'rho' for plotting embedding functions."
        )
        group.add_argument(
            '--rhot',
            type=float,
            default=None,
            help="The final 'rho' for plotting embedding functions."
        )
        super(ExportModelProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """
        The main function.
        """
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

            kwargs = {'r0': args.r0, 'rt': args.rt,
                      'rho0': args.rho0, 'rhot': args.rhot,
                      'nnl_max': args.nnl}

            manager = TrainingManager(configs, validate_tfrecords=False)
            manager.export(ckpt,
                           tag=step_tag,
                           use_ema_variables=(not args.no_ema),
                           mode=args.mode,
                           **kwargs)
        return func
