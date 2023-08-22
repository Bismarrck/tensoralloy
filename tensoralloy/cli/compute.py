#!coding=utf-8
"""
Command-line programs under the `compute` scope.
"""
from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import argparse
import glob
import os
import warnings

from os.path import dirname, join, basename, exists
from ase.build import bulk
from ase.io import read, write
from ase.units import GPa, kB
from matplotlib import pyplot as plt

from tensoralloy.utils import ModeKeys
from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.io.input.reader import InputReader
from tensoralloy.train.training import TrainingManager
from tensoralloy.analysis.elastic import get_lattice_type
from tensoralloy.analysis.elastic import get_elementary_deformations
from tensoralloy.analysis.elastic import get_elastic_tensor
from tensoralloy.analysis.eos import EquationOfState
from tensoralloy.cli.cli import CLIProgram
from tensoralloy.nn.constraint.data import read_external_crystal
from tensoralloy.nn.constraint.data import built_in_crystals
from tensoralloy.precision import precision_scope
from tensoralloy.utils import Defaults
from tensoralloy import atoms_utils

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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
            ComputeScatterProgram(),
            EquationOfStateProgram(),
            ComputeElasticTensorProgram(),
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


def _get_atoms(name_or_filename: str):
    """
    A helper function to initialize an `Atoms` given a name or a filename.
    """
    if name_or_filename in built_in_crystals:
        return built_in_crystals[name_or_filename].atoms
    try:
        atoms = bulk(name_or_filename)
    except Exception:
        try:
            if name_or_filename.endswith("toml"):
                atoms = read_external_crystal(name_or_filename).atoms
            else:
                atoms = read(name_or_filename, index=0)
        except Exception:
            raise ValueError(f"Unrecognized str: {name_or_filename}")
    return atoms


class ComputeElasticTensorProgram(CLIProgram):
    """
    Compute the elastic constants tensor of a crystal.
    """

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "elastic"

    @property
    def help(self):
        """
        The help message.
        """
        return "Compute the elastic constants tensor of a crystal."

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
            "graph_model_file",
            type=str,
            help="The graph model file to use."
        )
        subparser.add_argument(
            "--analytic",
            default=False,
            action="store_true",
            help="Compute the elastic constants analytically if possible."
        )

        group = subparser.add_argument_group("Numeric Options")
        group.add_argument(
            "--n",
            type=int,
            default=10,
            help="Set the number of deformed structures. Defaults to 10.",
        )
        group.add_argument(
            "--d",
            type=float,
            default=0.5,
            help="The deformation distance. Defaults to 0.5."
        )

        super(ComputeElasticTensorProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """
        The main function of this program.
        """
        def func(args: argparse.Namespace):
            crystal = _get_atoms(args.crystal)
            calc = TensorAlloyCalculator(args.graph_model_file)
            crystal.calc = calc

            if args.analytic and "elastic" in calc.implemented_properties:
                tensor = calc.get_elastic_constant_tensor(crystal)
                lattyp, brav, sg_name, sg_nr = get_lattice_type(crystal)
            else:
                systems = get_elementary_deformations(
                    crystal, n=args.n, d=args.d)
                for atoms in systems:
                    atoms.calc = calc
                tensor, _, lattice = get_elastic_tensor(crystal, systems)
                lattyp = lattice['lattice_number']
                brav = lattice['lattice_name']
                sg_name = lattice['spacegroup']
                sg_nr = lattice['spacegroup_number']

            np.set_printoptions(precision=0, suppress=True)

            print(f"Crystal: {args.crystal}")
            print(f"Lattice: {brav}({lattyp})")
            print(f"SpaceGroup: {sg_name}({sg_nr})")
            print("The elastic constants tensor: ")
            print(tensor.voigt)

        return func


class ComputeScatterProgram(CLIProgram):
    """
    Compute MAEs with respect to a metric.
    """

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "scatter"

    @property
    def help(self):
        """
        The help message.
        """
        return "Compute MAEs with respect to (P, T) using a checkpoint"

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
            '--versus',
            choices=['fmax', 'fnorm'],
            default='fmax',
            help="The target property"
        )
        subparser.add_argument(
            '--tf-records-dir',
            type=str,
            default='..',
            help="The directory where tfrecord files should be found."
        )
        subparser.add_argument(
            '--xscale',
            type=str,
            choices=['linear', 'log'],
            default='linear',
            help="The x-axis scale."
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

        super(ComputeScatterProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            model_dir = dirname(args.ckpt)
            candidates = list(glob.glob(join(model_dir, "*.toml")))
            if len(candidates) > 1:
                warnings.warn(f"More than one TOML file in {model_dir}. "
                              f"Only {candidates[0]} will be used.")
            filename = candidates[0]

            config = InputReader(filename)
            config['dataset.sqlite3'] = basename(config['dataset.sqlite3'])
            config['dataset.tfrecords_dir'] = args.tf_records_dir
            config['nn.minimize'] = ('energy', 'forces', 'stress')
            precision = config['precision']

            with precision_scope(precision):
                with tf.Graph().as_default():

                    manager = TrainingManager(config, validate_tfrecords=True)
                    if args.use_train_data:
                        mode = ModeKeys.TRAIN
                        size = manager.dataset.train_size
                        batch_size = manager.hparams.train.batch_size
                    else:
                        mode = ModeKeys.EVAL
                        size = manager.dataset.test_size
                        batch_size = manager.hparams.train.batch_size
                        batch_size = min(size, batch_size)
                        if size % batch_size != 0:
                            batch_size = 1
                    natoms = sum(manager.dataset.max_occurs.values())

                    input_fn = manager.dataset.input_fn(
                        mode=mode,
                        batch_size=batch_size,
                        num_epochs=1,
                        shuffle=False
                    )
                    features, labels = input_fn()
                    predictions = manager.model.build(features=features,
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

                        true_vals = {
                            'energy': np.zeros(size, dtype=float),
                            'stress': np.zeros((size, 3), dtype=float),
                            'f_norm': np.zeros(size, dtype=float),
                            'f_max': np.zeros(size, dtype=float)
                        }
                        pred_vals = {
                            'energy': np.zeros(size, dtype=float),
                            'stress': np.zeros((size, 3), dtype=float),
                        }

                        istart = 0
                        for _ in tqdm.trange(size // batch_size):
                            predictions_, labels_, n_atoms_, mask_ = sess.run([
                                predictions,
                                labels,
                                features["n_atoms_vap"],
                                features["atom_masks"]])
                            mask_ = mask_[:, 1:].astype(bool)
                            iend = istart + batch_size

                            for prop in config['nn.minimize']:
                                if prop == 'energy':
                                    true_vals[prop][istart: iend] = \
                                        labels_[prop] / n_atoms_
                                    pred_vals[prop][istart: iend] = \
                                        predictions_[prop] / n_atoms_
                                elif prop == 'forces':
                                    f_true = labels_[prop][:, 1:, :]
                                    true_vals['f_norm'][istart: iend] = \
                                        np.linalg.norm(f_true, axis=(1, 2)) / \
                                            n_atoms_**0.5
                                    true_vals['f_max'][istart: iend] = \
                                        np.max(np.abs(f_true), axis=(1, 2))
                                else:
                                    true_vals[prop][istart: iend, :3] = \
                                        labels_[prop][:, :3] / GPa
                                    pred_vals[prop][istart: iend, :3] = \
                                        predictions_[prop][:, :3] / GPa
                
                mae = {
                    'energy': np.abs(
                        true_vals['energy'] - pred_vals['energy']),
                    'stress': np.mean(
                        np.abs(true_vals['stress'] - pred_vals['stress']), 
                        axis=1)
                    }
                
                if args.versus == 'fmax':
                    key = 'f_max'
                else:
                    key = 'f_norm'

                _, axes = plt.subplots(1, 2, figsize=[10, 4])
                ax = axes[0]
                ax.plot(true_vals['f_max'], mae['energy'], 'k.') 
                ax.set_xlabel(r'$F_{max}$ (eV/$\AA$)', fontsize=16)
                ax.set_ylabel(r'MAE (eV/atom)', fontsize=16)

                if args.xscale == 'log':
                    ax.set_xscale('log')

                ax = axes[1]
                ax.plot(true_vals['f_max'], mae['stress'], 'k.') 
                ax.set_xlabel(r'$F_{max}$ (eV/$\AA$)', fontsize=16)
                ax.set_ylabel(r'Stress (xx,yy,zz) (GPa)', fontsize=16)

                if args.xscale == 'log':
                    ax.set_xscale('log')

                plt.tight_layout()
                plt.savefig(f"scatter_{key}.png", dpi=150)

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

            with precision_scope(precision):
                with tf.Graph().as_default():

                    manager = TrainingManager(config, validate_tfrecords=True)
                    if args.use_train_data:
                        mode = ModeKeys.TRAIN
                        size = manager.dataset.train_size
                        batch_size = manager.hparams.train.batch_size
                    else:
                        mode = ModeKeys.EVAL
                        size = manager.dataset.test_size
                        batch_size = manager.hparams.train.batch_size
                        batch_size = min(size, batch_size)
                        if size % batch_size != 0:
                            batch_size = 1
                    n_used = size - divmod(size, batch_size)[1]

                    input_fn = manager.dataset.input_fn(
                        mode=mode,
                        batch_size=batch_size,
                        num_epochs=1,
                        shuffle=False
                    )
                    features, labels = input_fn()
                    predictions = manager.model.build(features=features,
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
                                features["n_atoms_vap"],
                                features["atom_masks"]])
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
                            rbound = int(n_used * np.round(q / 100.0, 2)) - 1
                            reordered[prop].append(
                                np.mean(new_abs_diff[:rbound]))

                    reordered[order] = []
                    true_vals[key] = np.asarray(true_vals[key])[indices]
                    for q in range(args.q, 101, args.q):
                        rbound = int(n_used * np.round(q / 100.0, 2)) - 1
                        reordered[order].append(true_vals[key][rbound])

                    dataframe = pd.DataFrame(reordered)
                    dataframe.set_index(order, inplace=True)

                # Convert to meV/atoms
                dataframe['energy'] *= 1000.0

                name_mapping = {
                    'energy': 'Energy (meV/atom)',
                    'forces': 'Force (eV/Ang)',
                    'stress': 'Stress (GPa)',
                    'total_pressure': 'Pressure (GPa)'
                }
                rename_cols = {old: new for old, new in name_mapping.items()
                               if old in dataframe.columns.to_list()}
                dataframe.rename(columns=rename_cols, inplace=True)

                pd.options.display.float_format = "{:.6f}".format
                print(f"Mode: {mode} ({n_used}/{size})")
                print(dataframe.to_string())

                with open(f"percentile.{args.ckpt.replace('.', '_')}.dat", "w") as f:
                    f.write(f"Mode: {mode} ({n_used}/{size})\n")
                    f.write(dataframe.to_string())
                    f.write("\n")

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
                     'antonschmidt', 'rose'],
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
            "--dx",
            type=float,
            default=0.01,
            help="The scaling step"
        )
        subparser.add_argument(
            "--beta",
            type=float,
            default=0.005,
            help="The parameter beta for Rose Universal EoS"
        )
        subparser.add_argument(
            '--fig',
            type=str,
            default=None,
            help="The output volume-energy figure."
        )
        subparser.add_argument(
            '--etemp',
            type=float,
            default=0.0,
            help="The electron temperature in Kelvin."
        )
        subparser.add_argument(
            '--pv',
            action="store_true",
            default=False,
        )
        subparser.add_argument(
            '--use-free-energy',
            action='store_true',
            default=False,
            help="Plot free energy vs volume if this flag is set."
        )

        super(EquationOfStateProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            assert 0.40 <= args.xlo
            assert args.xlo < args.xhi <= 1.25
            assert 0.00 <= args.etemp

            crystal = _get_atoms(args.crystal)
            atoms_utils.set_electron_temperature(crystal, args.etemp * kB)
            calc = TensorAlloyCalculator(args.graph_model_path)
            crystal.calc = calc
            cell = crystal.cell.copy()
            formula = crystal.get_chemical_formula()
            properties = ['energy', 'stress']
            label = "E"
            fit_tag = "eos"
            if args.pv:
                fit_tag = "PV"
            if args.etemp > 0:
                formula = f"{formula}_{args.etemp:.0f}K"
                properties.extend(['eentropy', 'free_energy'])
            if args.fig is None:
                work_dir = "."
                figname = join(work_dir, f"{formula}_{fit_tag}.png")
                csvname = join(work_dir, f"{formula}_{fit_tag}.csv")
            else:
                work_dir = dirname(args.fig)
                figname = args.fig
                csvname = f"{basename(figname)}_{fit_tag}.csv"
                if work_dir and not exists(work_dir):
                    os.makedirs(work_dir)

            volumes = []
            energies = []
            eentropies = []
            num = int(np.round(args.xhi - args.xlo, 2) / args.dx) + 1
            for x in np.linspace(args.xlo, args.xhi, num, True):
                crystal.set_cell(cell * x, scale_atoms=True)
                volumes.append(crystal.get_volume())
                calc.calculate(crystal, properties=properties)
                if args.pv:
                    energies.append(-calc.results['stress'][:3].mean() / GPa)
                else:
                    if args.use_free_energy:
                        energies.append(calc.results['free_energy'])
                        label = "F"
                    else:
                        energies.append(calc.results['energy'])
                if args.etemp > 0:
                    eentropies.append(calc.results['eentropy'])
                else:
                    eentropies.append(0.0)

            eos = EquationOfState(volumes, energies, eos=args.eos,
                                  beta=args.beta)
            v0, e0, bulk_modulus, residual = eos.fit()
            eos.plot(figname, show=False, pv=args.pv)

            if not args.pv:
                df = pd.DataFrame({label: energies,
                                   "V": volumes,
                                   "S": eentropies,
                                   "T(K)": np.ones_like(energies) * args.etemp})
                df.to_csv(csvname)

                print("{}/{}, V0 = {:.3f}, E0 = {:.3f} eV, B = {} GPa".format(
                    formula,
                    args.eos,
                    v0, e0, bulk_modulus))
                print("Residual Norm = {:.3f} eV".format(residual))

                np.set_printoptions(precision=3, suppress=True)
                new_cell = (v0 / np.linalg.det(cell))**(1.0/3.0) * cell
                print("New cell: ")
                print(new_cell)

                eq = crystal.copy()
                eq.set_cell(new_cell, True)
                write(f"{formula}_Eq.extxyz", eq)

        return func
