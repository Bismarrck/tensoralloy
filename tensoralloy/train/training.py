#!coding=utf-8
"""
This module is used to train the model
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import shutil
import os
import platform
import logging

from os.path import join, exists, dirname, basename, realpath
from typing import Union
from tensorflow.python import debug as tf_debug
from tensorflow_estimator import estimator as tf_estimator
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from tensoralloy.train.dataset import Dataset
from tensoralloy.io.input import InputReader
from tensoralloy.io.db import connect
from tensoralloy.nn.atomic.sf import SymmetryFunctionNN
from tensoralloy.nn.atomic.deepmd import DeepPotSE
from tensoralloy.nn.atomic.grap import GenericRadialAtomicPotential
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.nn.eam.fs import EamFsNN
from tensoralloy.nn.eam.adp import AdpNN
from tensoralloy.nn.tersoff import Tersoff
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.transformer.universal import BatchUniversalTransformer
from tensoralloy.utils import set_logging_configs, nested_set
from tensoralloy.utils import check_path
from tensoralloy.precision import precision_scope
from tensoralloy.train.dataclasses import EstimatorHyperParams


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TrainingManager:
    """
    This class is used to train/evaluate/export a `BasicNN` model initialized
    from a TOML input file.
    """

    def __init__(self,
                 input_file: Union[str, InputReader],
                 validate_tfrecords=True):
        """
        Initialization method.

        Parameters
        ----------
        input_file : str or InputReader
            The input TOML file to read or an `InputReader`.
        validate_tfrecords : bool
            If True, the corresponding tfrecords files will be created if
            missing.

        """
        if isinstance(input_file, str):
            self._reader = InputReader(input_file)
        elif isinstance(input_file, InputReader):
            self._reader = input_file
        else:
            raise ValueError("`input_file` should be a str or InputReader!")

        self._float_precision = self._reader['precision']

        with precision_scope(self._float_precision):
            self._dataset = self._get_dataset(validate_tfrecords)
            self._hparams = self._get_hparams()
            self._nn = self._get_nn()
            self._input_file = input_file

    @property
    def nn(self):
        """
        Return a `BasicNN`.
        """
        return self._nn

    @property
    def dataset(self) -> Dataset:
        """
        Return the dataset.
        """
        return self._dataset

    @property
    def hparams(self) -> EstimatorHyperParams:
        """
        Return a dict of hyper parameters.
        """
        return self._hparams

    def _backup_input_file(self):
        """
        Copy the input TOML file and the sqlite3 database  to `model_dir`.
        """
        files = (self._input_file, self._reader['dataset.sqlite3'])

        for src in files:
            dst = join(self._hparams.train.model_dir, basename(src))
            if realpath(dst) == realpath(src):
                dst += '.bak'
            if exists(dst):
                os.remove(dst)
            shutil.copyfile(src, dst)

    def _get_hparams(self):
        """
        Initialize the hyper parameters.
        """
        hparams = EstimatorHyperParams.from_input_reader(self._reader)

        if not hparams.opt.decay_function:
            hparams.opt.decay_function = None
        if not hparams.train.ckpt.checkpoint_filename:
            hparams.train.ckpt.checkpoint_filename = None

        if self._dataset.test_size % hparams.train.batch_size != 0:
            eval_batch_size = next(
                x for x in range(hparams.train.batch_size, 0, -1)
                if self._dataset.test_size % x == 0)
            print(f"Warning: batch_size is reduced to {eval_batch_size} "
                  f"for evaluation")
        else:
            eval_batch_size = hparams.train.batch_size
        hparams.train.eval_batch_size = eval_batch_size

        return hparams

    def _get_atomic_nn(self, kwargs: dict):
        """
        Initialize an atomistic neural network potential.
        """
        hidden_sizes = {}
        for element in kwargs['elements']:
            keypath = f'nn.atomic.layers.{element}'
            value = self._reader[keypath]
            if value is not None:
                hidden_sizes[element] = value

        pair_style = self._reader['pair_style']
        configs = self._reader['nn.atomic']
        params = {
            'activation': configs['activation'],
            'hidden_sizes': hidden_sizes,
            'kernel_initializer': configs['kernel_initializer'],
            'use_atomic_static_energy': configs['use_atomic_static_energy'],
            'fixed_atomic_static_energy': configs['fixed_atomic_static_energy'],
            'atomic_static_energy': self._dataset.atomic_static_energy,
            'use_resnet_dt': configs['use_resnet_dt'],
            'finite_temperature': configs['finite_temperature'],
        }
        params.update(kwargs)

        if pair_style == 'atomic/sf':
            for key in ('eta', 'omega', 'gamma', 'zeta', 'beta',
                        'cutoff_function', 'minmax_scale'):
                params[key] = configs['sf'][key]
            return SymmetryFunctionNN(**params)
        elif pair_style == 'atomic/deepmd':
            params.update(configs['deepmd'])
            return DeepPotSE(**params)
        else:
            for key in ('moment_tensors',
                        'algorithm',
                        'cutoff_function',
                        'param_space_method'):
                params[key] = configs['grap'][key]
            params['parameters'] = configs['grap'][params['algorithm']]
            return GenericRadialAtomicPotential(**params)

    def _get_eam_nn(self, kwargs: dict) -> Union[EamAlloyNN, EamFsNN, AdpNN]:
        """
        Initialize an `EamAlloyNN` or an 'EamFsNN'.
        """

        hidden_sizes = {}
        custom_potentials = {}

        for pot in ('rho', 'embed', 'phi', 'dipole', 'quadrupole'):
            keypath = f'nn.eam.{pot}'
            if self._reader[keypath] is None:
                continue
            for key in self._reader[keypath].keys():
                value = self._reader[f'nn.eam.{pot}.{key}']
                if value is None:
                    continue
                if isinstance(value, str):
                    if value not in available_potentials:
                        if not value.startswith("spline@"):
                            raise ValueError(
                                f"The empirical potential "
                                f"[{pot}.{value}] is not available")
                    nested_set(custom_potentials, f'{key}.{pot}', value)
                else:
                    nested_set(hidden_sizes, f'{key}.{pot}', value)
                    nested_set(custom_potentials, f'{key}.{pot}', 'nn')

        kwargs.update(dict(hidden_sizes=hidden_sizes,
                           custom_potentials=custom_potentials))

        pair_style = self._reader['pair_style']
        if pair_style == "eam/alloy":
            return EamAlloyNN(**kwargs)
        elif pair_style == "eam/fs":
            return EamFsNN(**kwargs)
        elif pair_style == "eam/adp":
            return AdpNN(**kwargs)
        else:
            raise ValueError(f"Unknown pair_style {pair_style}")

    def _get_tersoff_nn(self, kwargs: dict) -> Tersoff:
        """
        Initialize a `Tersoff` model.
        """
        symmetric_mixing = self._reader['nn.tersoff.symmetric_mixing']
        potential_file = self._reader['nn.tersoff.file']
        kwargs.update(dict(symmetric_mixing=symmetric_mixing,
                           custom_potentials=potential_file))
        return Tersoff(**kwargs)

    def _get_nn(self):
        """
        Initialize a `BasicNN`.
        """
        elements = self._dataset.transformer.elements
        minimize_properties = self._reader['nn.minimize']
        export_properties = self._reader['nn.export']
        kwargs = {'elements': elements,
                  'minimize_properties': minimize_properties,
                  'export_properties': export_properties}
        pair_style = self._reader['pair_style']
        if pair_style.startswith("atomic"):
            nn = self._get_atomic_nn(kwargs)
        elif pair_style == "tersoff":
            nn = self._get_tersoff_nn(kwargs)
        else:
            nn = self._get_eam_nn(kwargs)
        nn.attach_transformer(self._dataset.transformer)
        return nn

    def _get_dataset(self, validate_tfrecords=True):
        """
        Initialize a `Dataset` using the configs of the input file.
        """
        database = connect(self._reader['dataset.sqlite3'])

        pair_style = self._reader['pair_style']
        rcut = self._reader['rcut']
        acut = self._reader['acut']

        max_occurs = database.max_occurs
        nij_max = database.get_nij_max(rcut, allow_calculation=True)
        nnl_max = database.get_nnl_max(rcut, allow_calculation=True)
        angular = False
        angular_symmetricity = False
        ij2k_max = 0
        nijk_max = 0

        if pair_style == 'atomic/sf' and self._reader['nn.atomic.sf.angular']:
            angular = True
            angular_symmetricity = True
            nijk_max = database.get_nijk_max(acut, allow_calculation=True)
            ij2k_max = database.get_ij2k_max(acut, allow_calculation=True)

        elif pair_style == "tersoff":
            nijk_max = database.get_nijk_max(
                acut, allow_calculation=True, symmetric=False)
            ij2k_max = database.get_ij2k_max(acut, allow_calculation=True)
            angular = True
            angular_symmetricity = False

        clf = BatchUniversalTransformer(
            max_occurs=max_occurs, rcut=rcut, angular=angular, nij_max=nij_max,
            nnl_max=nnl_max, nijk_max=nijk_max, ij2k_max=ij2k_max,
            symmetric=angular_symmetricity, use_forces=database.has_forces,
            use_stress=database.has_stress)

        name = self._reader['dataset.name']
        serial = self._reader['dataset.serial']
        dataset = Dataset(database=database, transformer=clf,
                          name=name, serial=serial)

        test_size = self._reader['dataset.test_size']
        tfrecords_dir = self._reader['dataset.tfrecords_dir']
        if not dataset.load_tfrecords(tfrecords_dir, test_size=test_size):
            if validate_tfrecords:
                dataset.to_records(
                    tfrecords_dir, test_size=test_size, verbose=True)
            else:
                tf.logging.info("Warning: tfrecords files missing.")
        return dataset

    @staticmethod
    def _check_before_training(hparams: EstimatorHyperParams):
        """
        Check the `model_dir` and `previous_checkpoint` before training.
        """
        model_dir = hparams.train.model_dir
        deleted = False

        if exists(model_dir):
            if hparams.train.reset_global_step:
                shutil.rmtree(model_dir, ignore_errors=True)
                tf.gfile.MakeDirs(hparams.train.model_dir)
                deleted = True
        else:
            tf.gfile.MakeDirs(hparams.train.model_dir)

        if hparams.train.ckpt.checkpoint_filename:
            ckpt_dir = dirname(hparams.train.ckpt.checkpoint_filename)
            if (realpath(ckpt_dir) == realpath(model_dir)) and deleted:
                print(f"Warning: {hparams.train.ckpt.checkpoint_filename} "
                      f"was already deleted")
                hparams.train.ckpt.checkpoint_filename = None

    @staticmethod
    def _get_logging_level(hparams: EstimatorHyperParams):
        """
        Return the logging level controlled by `hparams.debug.logging_level`.
        """
        levels = {
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
        }
        return levels.get(hparams.debug.logging_level.lower(), logging.INFO)

    def train_and_evaluate(self, debug=False):
        """
        Initialize a model and train it with `tf_estimator.train_and_evalutate`.
        """

        graph = tf.Graph()
        precision = self._float_precision

        with precision_scope(precision):
            with graph.as_default():

                dataset = self._dataset
                nn = self._nn
                hparams = self._hparams

                self._check_before_training(hparams)
                self._backup_input_file()

                set_logging_configs(
                    logfile=check_path(join(
                        hparams.train.model_dir, 'logfile')),
                    level=self._get_logging_level(hparams))

                tf.logging.info(f'pid={os.getpid()}')
                tf.logging.info(f'seed={self._hparams.seed}')
                tf.logging.info(f'input= \n{str(self._reader)}')

                if tf.__version__ >= "1.14":
                    from tensoralloy.train import distribute_utils
                    strategy = distribute_utils.get_distribution_strategy(
                        **hparams.distribute.as_dict()
                    )
                else:
                    strategy = None
                if strategy is None:
                    train_input_fn = dataset.input_fn(
                        mode=tf_estimator.ModeKeys.TRAIN,
                        batch_size=hparams.train.batch_size,
                        shuffle=hparams.train.shuffle)
                else:
                    # The lambda wrap of `input_fn` is necessary for distributed
                    # training.
                    train_input_fn = lambda: dataset.input_fn(
                        mode=tf_estimator.ModeKeys.TRAIN,
                        batch_size=hparams.train.batch_size,
                        shuffle=hparams.train.shuffle)

                timeout_ms = hparams.debug.meta_optimizer_timeout_ms
                session_config = tf.ConfigProto(
                    allow_soft_placement=hparams.debug.allow_soft_placement,
                    log_device_placement=hparams.debug.log_device_placement,
                    gpu_options=tf.GPUOptions(
                        allow_growth=hparams.debug.allow_gpu_growth),
                    graph_options=tf.GraphOptions(
                        rewrite_options=RewriterConfig(
                            meta_optimizer_timeout_ms=timeout_ms,
                        )
                    ))

                # TODO: set the evaluation strategy

                run_config = tf_estimator.RunConfig(
                    save_checkpoints_steps=hparams.train.eval_steps,
                    tf_random_seed=hparams.seed,
                    log_step_count_steps=None,
                    keep_checkpoint_max=hparams.train.max_checkpoints_to_keep,
                    train_distribute=strategy,
                    session_config=session_config)

                estimator = tf_estimator.Estimator(
                    model_fn=nn.model_fn,
                    warm_start_from=None,
                    model_dir=hparams.train.model_dir,
                    config=run_config,
                    params=hparams)

                if debug:
                    system = platform.system().lower()
                    if system == 'linux':
                        ui_type = 'curses'
                    else:
                        ui_type = 'readline'
                    hooks = [tf_debug.LocalCLIDebugHook(ui_type=ui_type), ]
                else:
                    hooks = None

                train_spec = tf_estimator.TrainSpec(
                    input_fn=train_input_fn,
                    max_steps=hparams.train.train_steps,
                    hooks=hooks)

                eval_spec = tf_estimator.EvalSpec(
                    input_fn=dataset.input_fn(
                        mode=tf_estimator.ModeKeys.EVAL,
                        batch_size=hparams.train.eval_batch_size,
                        num_epochs=1,
                        shuffle=False),
                    steps=hparams.train.eval_steps,
                    start_delay_secs=hparams.debug.start_delay_secs,
                    # Explicitly set these thresholds to lower values so that
                    # every checkpoint can be evaluated.
                    throttle_secs=hparams.debug.throttle_secs,
                )
                tf_estimator.train_and_evaluate(
                    estimator, train_spec, eval_spec)

    def export(self, checkpoint=None, tag=None, use_ema_variables=True,
               export_lammps_mpi_pb=False, **kwargs):
        """
        Export the trained model.
        """
        precision = self._float_precision
        with precision_scope(precision):
            if checkpoint is None:
                checkpoint = tf.train.latest_checkpoint(
                    self._hparams.train.model_dir)

            if export_lammps_mpi_pb:
                pb_ext = "lmpb"
            else:
                pb_ext = "pb"
            if tag is not None:
                graph_name = f'{self._dataset.name}.{tag}.{pb_ext}'
            else:
                graph_name = f'{self._dataset.name}.{pb_ext}'
            graph_path = join(self._hparams.train.model_dir, graph_name)

            self._nn.export(
                output_graph_path=graph_path,
                checkpoint=checkpoint,
                use_ema_variables=use_ema_variables,
                keep_tmp_files=False,
                export_partial_forces_model=export_lammps_mpi_pb)

            if isinstance(self._nn, (EamAlloyNN, EamFsNN, AdpNN)):
                setfl_kwargs = self._reader['nn.eam.setfl']

                if 'lattice' in setfl_kwargs:
                    lattice = setfl_kwargs.pop('lattice')
                    lattice_constants = lattice.get('constant', {})
                    lattice_types = lattice.get('type', {})
                else:
                    lattice_constants = None
                    lattice_types = None

                if isinstance(self._nn, AdpNN):
                    if tag is not None:
                        setfl = f'{self._dataset.name}.{tag}.adp'
                    else:
                        setfl = f'{self._dataset.name}.adp'
                else:
                    if tag is not None:
                        setfl = f'{self._dataset.name}.{self._nn.tag}.{tag}.eam'
                    else:
                        setfl = f'{self._dataset.name}.{self._nn.tag}.eam'

                self._nn.export_to_setfl(
                    setfl=join(self._hparams.train.model_dir, setfl),
                    checkpoint=checkpoint,
                    lattice_constants=lattice_constants,
                    lattice_types=lattice_types,
                    use_ema_variables=use_ema_variables,
                    **setfl_kwargs,
                    **kwargs)
