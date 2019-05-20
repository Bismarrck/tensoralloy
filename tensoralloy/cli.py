# coding=utf-8
"""
This module defines the command-line main function of `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import glob
import warnings
import re
import os

from ase.units import GPa
from ase.io import read
from ase.build import bulk
from os.path import exists, dirname, join, basename, splitext
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.analysis.eos import EquationOfState
from tensoralloy.io.read import read_file
from tensoralloy.io.input import InputReader
from tensoralloy.nn.constraint.data import read_external_crystal
from tensoralloy.test_utils import datasets_dir
from tensoralloy.train import TrainingManager
from tensoralloy.utils import Defaults
from tensoralloy.precision import set_precision
from tensoralloy.calculator import TensorAlloyCalculator

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
                      'rho0': args.rho0, 'rhot': args.rhot}

            manager = TrainingManager(configs, validate_tfrecords=False)
            manager.export(ckpt,
                           tag=step_tag,
                           use_ema_variables=(not args.no_ema),
                           **kwargs)
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
        """
        The main function.
        """
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


class PrintEvaluationSummaryProgram(CLIProgram):
    """
    Print the summary of all evaluations of an experiment.
    """

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "print"

    @property
    def help(self):
        """
        The help message.
        """
        return "Print the summary of the evaluation results."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'logfile',
            type=str,
            help="The logfile of an experiment."
        )

        super(PrintEvaluationSummaryProgram, self).config_subparser(subparser)

    @staticmethod
    def print_evaluation_summary(logfile) -> pd.DataFrame:
        """
        Summarize the evalutaion results of the logfile.
        """

        global_step_patt = re.compile(r".*tensorflow\s+INFO\s+Saving\sdict"
                                      r"\sfor\sglobal\sstep\s(\d+):(.*)")
        key_value_pair_patt = re.compile(r"\s+(.*)\s=\s([0-9.-]+)")
        pid_patt = re.compile(r".*tensorflow\s+INFO\s+pid=(\d+)")
        results = {}

        with open(logfile) as fp:
            for line in fp:
                line = line.strip()
                if pid_patt.search(line):
                    results.clear()
                    continue

                m = global_step_patt.search(line)
                if not m:
                    continue

                for s in m.group(2).split(','):
                    key_value_pair = key_value_pair_patt.search(s)
                    key = key_value_pair.group(1)
                    val = key_value_pair.group(2)

                    if key == 'global_step':
                        convert_fn = int

                    elif key.startswith('Elastic'):
                        def convert_fn(_x):
                            """ Convert the string to int. """
                            return "%.1f" % np.round(float(_x), 1)

                        if 'Constraints' in key:
                            key = key[8:].replace('/Constraints', '')
                        else:
                            key = key[8:].replace('/Cijkl', '')
                    else:
                        convert_fn = float

                    results[key] = results.get(key, []) + [convert_fn(val)]

        df = pd.DataFrame(results)
        df.set_index('global_step', inplace=True)

        print(df.to_string())

        with open(join(dirname(logfile), 'summary.csv'), 'w') as fp:
            fp.write(df.to_csv())

        return df

    @property
    def main_func(self):
        """
        The main function of this program.
        """
        def func(args: argparse.Namespace):
            logfile = args.logfile
            if not exists(logfile):
                raise IOError(f"The logfile {logfile} cannot be accessed!")
            self.print_evaluation_summary(logfile)
        return func


class ComputeMetricsProgram(CLIProgram):
    """
    A collection of programs for computing advanced evaluation metrics. 
    """
    
    def __init__(self):
        """
        Initiazliation method.
        """
        super(ComputeMetricsProgram, self).__init__()
        
        self._programs = [
            ComputeEvaluationPercentileProgram(),
            EquationOfStateProgram()
        ]
    
    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "compute"

    @property
    def help(self):
        """
        The help message
        """
        return "Compute advanced evaluation metrics."
    
    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the subparaser.
        """
        subsubparsers = subparser.add_subparsers(title=self.name,
                                                 help=self.help)
        for prog in self._programs:
            subsubparser = subsubparsers.add_parser(prog.name, help=prog.help)
            prog.config_subparser(subsubparser)
        super(ComputeMetricsProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """ The main function. This is just a program-collection, so this
        function is empty. """
        def func(_):
            pass
        return func


class ComputeEvaluationPercentileProgram(CLIProgram):
    """
    Compute the q-th percentile of per-atom MAEs of the selected checkpoint.
    """

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "percentile"

    @property
    def help(self):
        """
        The help message.
        """
        return "Compute q-th percentile of per-atom MAEs of the selected " \
               "checkpoint."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'ckpt',
            type=str,
            help="The checkpoint file to use."
        )
        subparser.add_argument(
            '--tf-records-dir',
            type=str,
            default='..',
            help="The directory where tfrecord files should be found."
        )
        subparser.add_argument(
            '--q',
            type=int,
            default=5,
            help="Percentile or sequence of percentiles to compute, "
                 "which must be between 0 and 100 exclusive."
        )
        subparser.add_argument(
            '--no-ema',
            action='store_true',
            default=False,
            help="If this flag is given, EMA variables will be disabled."
        )
        subparser.add_argument(
            '--use-train-data',
            default=False,
            action='store_true',
            help="Use training data instead of test data."
        )
        subparser.add_argument(
            '--order',
            type=str,
            choices=['f_norm', 'dE'],
            default=None,
            help="Print the errors w.r.t RMS of true forces."
        )

        super(ComputeEvaluationPercentileProgram, self).config_subparser(
            subparser)

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            """
            Compute the
            """
            if args.q <= 0 or args.q >= 100:
                raise ValueError("`q` must be an integer and 0 < q < 100.")

            model_dir = dirname(args.ckpt)
            candidates = list(glob.glob(join(model_dir, "*.toml")))
            if len(candidates) > 1:
                warnings.warn(f"More than one TOML file in {model_dir}. "
                              f"Only {candidates[0]} will be used.")
            filename = candidates[0]

            config = InputReader(filename)
            config['dataset.sqlite3'] = basename(config['dataset.sqlite3'])
            config['dataset.tfrecords_dir'] = args.tf_records_dir

            avail_properties = ('energy', 'forces', 'stress', 'total_pressure')
            properties = [x for x in config['nn.minimize']
                          if x in avail_properties]
            if args.order == 'f_norm':
                if 'forces' not in properties:
                    properties.append('forces')

            config['nn.minimize'] = properties
            precision = config['precision']

            with set_precision(precision):
                with tf.Graph().as_default():

                    manager = TrainingManager(config, validate_tfrecords=True)
                    if args.use_train_data:
                        mode = tf_estimator.ModeKeys.TRAIN
                        size = manager.dataset.train_size
                    else:
                        mode = tf_estimator.ModeKeys.EVAL
                        size = manager.dataset.test_size

                    batch_size = manager.hparams.train.batch_size
                    n_used = size - divmod(size, batch_size)[1]

                    input_fn = manager.dataset.input_fn(
                        mode=mode,
                        batch_size=batch_size,
                        num_epochs=1,
                        shuffle=False
                    )
                    features, labels = input_fn()
                    predictions = manager.nn.build(features=features,
                                                   mode=mode,
                                                   verbose=True)

                    if args.no_ema:
                        saver = tf.train.Saver(tf.trainable_variables())
                    else:
                        ema = tf.train.ExponentialMovingAverage(
                            Defaults.variable_moving_average_decay)
                        saver = tf.train.Saver(ema.variables_to_restore())

                    with tf.Session() as sess:
                        tf.global_variables_initializer().run()
                        saver.restore(sess, args.ckpt)

                        true_vals = {x: [] for x in properties}
                        true_vals['f_norm'] = []
                        pred_vals = {x: [] for x in properties}

                        for i in range(size // batch_size):
                            predictions_, labels_, n_atoms_, mask_ = sess.run([
                                predictions,
                                labels,
                                features.n_atoms,
                                features.mask])
                            mask_ = mask_[:, 1:].astype(bool)

                            for prop in properties:
                                if prop == 'energy':
                                    true_vals[prop].extend(
                                        labels_[prop] / n_atoms_)
                                    pred_vals[prop].extend(
                                        predictions_[prop] / n_atoms_)
                                elif prop == 'forces':
                                    f_true = labels_[prop][:, 1:, :]
                                    f_norm = np.linalg.norm(f_true, axis=(1, 2))
                                    f_pred = predictions_[prop]
                                    for j in range(batch_size):
                                        true_vals[prop].extend(
                                            f_true[j, mask_[j], :].flatten())
                                        pred_vals[prop].extend(
                                            f_pred[j, mask_[j], :].flatten())
                                        true_vals['f_norm'].append(f_norm[j])
                                elif prop == 'stress':
                                    true_vals[prop].extend(labels_[prop] / GPa)
                                    pred_vals[prop].extend(
                                        predictions_[prop] / GPa)
                                else:
                                    true_vals[prop].extend(labels_[prop])
                                    pred_vals[prop].extend(predictions_[prop])

                data = {x: [] for x in properties}
                data['percentile'] = []
                abs_diff = {}
                for prop in properties:
                    abs_diff[prop] = np.abs(
                        np.subtract(np.array(true_vals[prop]),
                                    np.array(pred_vals[prop])))

                if args.order is None:
                    for q in range(0, 101, args.q):
                        data['percentile'].append(q)
                        if q == 100:
                            data['percentile'].append('MAE')
                            data['percentile'].append('Median')
                        for prop in properties:
                            data[prop].append(np.percentile(abs_diff[prop], q))
                            if q == 100:
                                data[prop].append(np.mean(abs_diff[prop]))
                                data[prop].append(np.median(abs_diff[prop]))
                    dataframe = pd.DataFrame(data)
                    dataframe.set_index('percentile', inplace=True)

                else:
                    order = args.order
                    mapping = {'dE': 'energy', 'f_norm': 'f_norm'}
                    key = mapping[order]

                    if order == 'dE':
                        emin = min(true_vals[key])
                        true_vals[key] = np.asarray(true_vals[key]) - emin
                    else:
                        true_vals[key] = np.asarray(true_vals[key])
                    indices = np.argsort(true_vals[key])

                    reordered = {}
                    for prop in ('energy', 'total_pressure'):
                        if prop not in abs_diff:
                            continue
                        new_abs_diff = abs_diff[prop][indices]
                        reordered[prop] = []
                        for q in range(args.q, 101, args.q):
                            reordered[prop].append(np.mean(new_abs_diff[:q]))

                    reordered[order] = []
                    true_vals[key] = np.asarray(true_vals[key])[indices]
                    total_size = len(true_vals[key])
                    for q in range(args.q, 101, args.q):
                        rbound = int(total_size * np.round(q / 100.0, 2)) - 1
                        reordered[order].append(true_vals[key][rbound])

                    dataframe = pd.DataFrame(reordered)
                    dataframe.set_index(order, inplace=True)

                dataframe.rename(
                    columns={'energy': 'energy/atom'}, inplace=True)
                pd.options.display.float_format = "{:.6f}".format
                print(f"Mode: {mode} ({n_used}/{size})")
                print(dataframe.to_string())

        return func


class EquationOfStateProgram(CLIProgram):
    """
    A program to compute EOS of a crystal.
    """

    @property
    def name(self):
        """
        The name of this program.
        """
        return "eos"

    @property
    def help(self):
        """
        The help message.
        """
        return "Compute the Equation of State of a crystal with a graph model."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'crystal',
            type=str,
            help="The name or filename of the target crystal."
        )
        subparser.add_argument(
            'graph_model_path',
            type=str,
            help="The graph model file to use."
        )
        subparser.add_argument(
            '--eos',
            type=str,
            default="birchmurnaghan",
            choices=['birchmurnaghan', 'birch', 'murnaghan', 'sj',
                     'pouriertarantola', 'vinet', 'p3', 'taylor',
                     'antonschmidt'],
            help="The equation to use.",
        )
        subparser.add_argument(
            '--xlo',
            type=float,
            default=0.95,
            help="The lower bound to scale the cell."
        )
        subparser.add_argument(
            '--xhi',
            type=float,
            default=1.05,
            help="The upper bound to scale the cell."
        )
        subparser.add_argument(
            '--fig',
            type=str,
            default=None,
            help="The output volume-energy figure."
        )

        super(EquationOfStateProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            assert 0.80 <= args.xlo <= 0.99
            assert 1.01 <= args.xhi <= 1.20

            calc = TensorAlloyCalculator(args.graph_model_path)

            try:
                crystal = bulk(args.crystal)
            except Exception:
                try:
                    if args.crystal.endswith("toml"):
                        crystal = read_external_crystal(args.crystal).atoms
                    else:
                        crystal = read(args.crystal, index=0)
                except Exception:
                    raise ValueError(f"Unrecognized {args.crystal}")
            crystal.calc = calc
            cell = crystal.cell.copy()
            formula = crystal.get_chemical_formula()

            if args.fig is None:
                work_dir = "."
                filename = join(work_dir, f"{formula}_eos.png")
            else:
                work_dir = dirname(args.fig)
                filename = args.fig
                if not exists(work_dir):
                    os.makedirs(work_dir)

            volumes = []
            energies = []
            num = int(np.round(args.xhi - args.xlo, 2) / 0.005) + 1
            for x in np.linspace(args.xlo, args.xhi, num, True):
                crystal.set_cell(cell * x, scale_atoms=True)
                volumes.append(crystal.get_volume())
                energies.append(crystal.get_potential_energy())

            eos = EquationOfState(volumes, energies, eos=args.eos)
            v0, e0, bulk_modulus, residual = eos.fit()
            eos.plot(filename, show=False)

            print("{}/{}, V0 = {:.3f}, E0 = {:.3f} eV, B = {} GPa".format(
                formula,
                args.eos,
                v0, e0, bulk_modulus))
            print("Residual Norm = {:.3f} eV".format(residual))

            np.set_printoptions(precision=3, suppress=True)
            print("New cell: ")
            print((v0 / np.linalg.det(cell))**(1.0/3.0) * cell)

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
                 StopExperimentProgram(),
                 PrintEvaluationSummaryProgram(),
                 ComputeMetricsProgram()):
        subparser = subparsers.add_parser(prog.name, help=prog.help)
        prog.config_subparser(subparser)

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
    else:
        args.func(args)
