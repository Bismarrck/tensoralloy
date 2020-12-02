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
import warnings

from os.path import join, exists, dirname, basename, realpath
from typing import Union
from tensorflow.python import debug as tf_debug
from tensorflow_estimator import estimator as tf_estimator
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from tensoralloy.train.dataset.dataset import PolarDataset, Dataset
from tensoralloy.io.input import InputReader
from tensoralloy.io.db import connect
from tensoralloy.nn.atomic import TemperatureDependentAtomicNN, AtomicNN
from tensoralloy.nn.atomic.sf import SymmetryFunction
from tensoralloy.nn.atomic.deepmd import DeepPotSE
from tensoralloy.nn.atomic.grap import GenericRadialAtomicPotential
from tensoralloy.nn.atomic.grap import GRAP_algorithms
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.nn.eam.fs import EamFsNN
from tensoralloy.nn.eam.adp import AdpNN
from tensoralloy.nn.tersoff import Tersoff
from tensoralloy.nn.atomic import special
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.transformer.universal import BatchUniversalTransformer
from tensoralloy.transformer.polar import BatchPolarTransformer
from tensoralloy.utils import set_logging_configs, nested_set
from tensoralloy.utils import check_path
from tensoralloy.precision import precision_scope
from tensoralloy.train.dataclasses import EstimatorHyperParams


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class PairStyle:
    """
    The pair style.
    """

    def __init__(self, pair_style: str):
        """
        Initialization.
        """
        self._pair_style = pair_style
        self._angular = False
        self._td = False
        self._special = False

        keys = pair_style.split("/")
        if keys[0] == 'special':
            if keys[1] == 'Be':
                self._td = True
                self._model = "grap"
                self._angular = False
                self._category = pair_style
            elif keys[1] == 'polar':
                self._td = False
                self._model = "grap"
                self._angular = False
                self._category = pair_style
            self._special = True
        else:
            self._category = keys[0]
            self._model = keys[1] if len(keys) > 1 else pair_style
            if len(keys) == 1:
                assert pair_style == "tersoff"
                self._angular = True
                self._td = False
                self._model = pair_style
            elif len(keys) == 3:
                assert keys[1] == 'sf' and keys[2] == 'angular'
                self._angular = True
                self._td = keys[0] == 'td'

    def __eq__(self, other):
        return self._pair_style == other

    def __str__(self):
        return self._pair_style

    @property
    def temperature_dependent(self):
        """ Return True if the pair style is temperature-depenedent. """
        return self._td

    @property
    def category(self):
        """ Return the category of the pair style (atomic, td, tersoff, eam) """
        return self._category

    @property
    def model(self):
        """
        Return the name of the potential of the pair style (sf, adp, eam, etc).
        """
        return self._model

    @property
    def angular(self):
        """ Return True if angular interactions should be used. """
        return self._angular

    @property
    def special(self):
        """ Return True if this is a special pair style. """
        return self._special


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
        self._pair_style = PairStyle(self._reader["pair_style"])

        with precision_scope(self._float_precision):
            self._dataset = self._get_dataset(validate_tfrecords)
            self._hparams = self._get_hparams()
            self._model = self._get_model()
            self._input_file = input_file

    @property
    def model(self):
        """
        Return a `BasicNN`.
        """
        return self._model

    @property
    def pair_style(self):
        """ Return the corresponding pair style. """
        return self._pair_style

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
        elements = kwargs['elements']
        hidden_sizes = {}
        for element in kwargs['elements']:
            keypath = f'nn.atomic.layers.{element}'
            value = self._reader[keypath]
            if value is not None:
                hidden_sizes[element] = value

        configs = self._reader['nn.atomic']
        params = {
            'activation': configs['activation'],
            'hidden_sizes': hidden_sizes,
            'kernel_initializer': configs['kernel_initializer'],
            'use_atomic_static_energy': configs['use_atomic_static_energy'],
            'fixed_atomic_static_energy': configs['fixed_atomic_static_energy'],
            'atomic_static_energy': self._dataset.atomic_static_energy,
            'minmax_scale': configs['minmax_scale'],
            'use_resnet_dt': configs['use_resnet_dt']
        }
        params.update(kwargs)

        if self._pair_style.model == 'sf':
            descriptor = SymmetryFunction(elements, **configs["sf"])
        elif self._pair_style.model == 'deepmd':
            descriptor = DeepPotSE(elements, **configs['deepmd'])
        else:
            algo = configs["grap"]["algorithm"]
            grap_kwargs = configs["grap"]
            grap_kwargs["parameters"] = configs["grap"][algo]
            for key in GRAP_algorithms:
                if key in grap_kwargs:
                    grap_kwargs.pop(key)
            descriptor = GenericRadialAtomicPotential(elements, **grap_kwargs)

        if self._pair_style.category == "td":
            cls = TemperatureDependentAtomicNN
            params['finite_temperature'] = configs['finite_temperature']
        elif self._pair_style == "special/Be":
            cls = special.BeNN
            params['finite_temperature'] = configs['finite_temperature']
        elif self._pair_style == "special/polar":
            cls = special.PolarNN
            params['inner_layers'] = configs['polar']['inner_layers']
            params['polar_loss_weight'] = configs['polar']['polar_loss_weight']
        else:
            cls = AtomicNN
        params["descriptor"] = descriptor
        return cls(**params)

    def _get_eam_nn(self, kwargs: dict) -> Union[EamAlloyNN, EamFsNN, AdpNN]:
        """
        Initialize an `EamAlloyNN` or an 'EamFsNN'.
        """

        hidden_sizes = {}
        custom_potentials = {}
        fixed_functions = self._reader["nn.eam.fixed_functions"]

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
                           custom_potentials=custom_potentials,
                           fixed_functions=fixed_functions))

        if self._pair_style.model == "alloy":
            return EamAlloyNN(**kwargs)
        elif self._pair_style.model == "fs":
            return EamFsNN(**kwargs)
        elif self._pair_style.model == "adp":
            return AdpNN(**kwargs)
        else:
            raise ValueError(f"Unknown pair_style {self._pair_style}")

    def _get_tersoff_nn(self, kwargs: dict) -> Tersoff:
        """
        Initialize a `Tersoff` model.
        """
        symmetric_mixing = self._reader['nn.tersoff.symmetric_mixing']
        potential_file = self._reader['nn.tersoff.file']
        kwargs.update(dict(symmetric_mixing=symmetric_mixing,
                           custom_potentials=potential_file))
        return Tersoff(**kwargs)

    def _get_model(self):
        """
        Initialize a `BasicNN`.
        """
        elements = self._dataset.transformer.elements
        minimize_properties = self._reader['nn.minimize']
        export_properties = self._reader['nn.export']
        kwargs = {'elements': elements,
                  'minimize_properties': minimize_properties,
                  'export_properties': export_properties}
        if self._pair_style == "tersoff":
            nn = self._get_tersoff_nn(kwargs)
        elif self._pair_style.category == "eam":
            nn = self._get_eam_nn(kwargs)
        else:
            nn = self._get_atomic_nn(kwargs)
        nn.attach_transformer(self._dataset.transformer)
        return nn

    def _get_dataset(self, validate_tfrecords=True):
        """
        Initialize a `Dataset` using the configs of the input file.
        """
        database = connect(self._reader['dataset.sqlite3'])

        rcut = self._reader['rcut']
        acut = self._reader['acut']

        max_occurs = database.max_occurs
        nij_max = database.get_nij_max(rcut, allow_calculation=True)
        nnl_max = database.get_nnl_max(rcut, allow_calculation=True)
        angular = self._pair_style.angular
        angular_symmetricity = self._pair_style != "tersoff"
        if angular:
            nijk_max = database.get_nijk_max(acut, allow_calculation=True)
            ij2k_max = database.get_ij2k_max(acut, allow_calculation=True)
        else:
            ij2k_max = 0
            nijk_max = 0

        if self._pair_style == "special/polar":
            cls = BatchPolarTransformer
        else:
            cls = BatchUniversalTransformer
        clf = cls(
            max_occurs=max_occurs, rcut=rcut, angular=angular, nij_max=nij_max,
            nnl_max=nnl_max, nijk_max=nijk_max, ij2k_max=ij2k_max,
            symmetric=angular_symmetricity, use_forces=database.has_forces,
            use_stress=database.has_stress)

        name = self._reader['dataset.name']
        serial = self._reader['dataset.serial']

        if self._pair_style == "special/polar":
            cls = PolarDataset
        else:
            cls = Dataset
        dataset = cls(database=database, transformer=clf, name=name,
                      serial=serial)

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
        warnings.filterwarnings("ignore")

        with precision_scope(precision):
            with graph.as_default():

                dataset = self._dataset
                nn = self._model
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

            self._model.export(
                output_graph_path=graph_path,
                checkpoint=checkpoint,
                use_ema_variables=use_ema_variables,
                keep_tmp_files=False,
                to_lammps=export_lammps_mpi_pb)

            if isinstance(self._model, (EamAlloyNN, EamFsNN, AdpNN)):
                setfl_kwargs = self._reader['nn.eam.setfl']

                if 'lattice' in setfl_kwargs:
                    lattice = setfl_kwargs.pop('lattice')
                    lattice_constants = lattice.get('constant', {})
                    lattice_types = lattice.get('type', {})
                else:
                    lattice_constants = None
                    lattice_types = None

                if isinstance(self._model, AdpNN):
                    if tag is not None:
                        setfl = f'{self._dataset.name}.{tag}.adp'
                    else:
                        setfl = f'{self._dataset.name}.adp'
                else:
                    if tag is not None:
                        setfl = f'{self._dataset.name}.{self._model.tag}.{tag}.eam'
                    else:
                        setfl = f'{self._dataset.name}.{self._model.tag}.eam'

                self._model.export_to_setfl(
                    setfl=join(self._hparams.train.model_dir, setfl),
                    checkpoint=checkpoint,
                    lattice_constants=lattice_constants,
                    lattice_types=lattice_types,
                    use_ema_variables=use_ema_variables,
                    **setfl_kwargs,
                    **kwargs)
