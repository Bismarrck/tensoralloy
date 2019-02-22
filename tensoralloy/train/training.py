# coding=utf-8
"""
This module is used to train the model
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import shutil
import os

from argparse import ArgumentParser, Namespace
from os.path import join, exists, dirname, basename, realpath
from ase.db import connect
from typing import Union
from tensorflow.python import debug as tf_debug

from tensoralloy.dataset import Dataset
from tensoralloy.io.input import InputReader
from tensoralloy.io.input.reader import nested_set
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn import EamFsNN, EamAlloyNN, AtomicNN, AtomicResNN
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.utils import set_logging_configs, AttributeDict
from tensoralloy.utils import check_path
from tensoralloy.dtypes import set_float_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TrainingManager:
    """
    This class is used to train/evaluate/export a `BasicNN` model initialized
    from a TOML input file.
    """

    def __init__(self, input_file: str, validate_tfrecords=True):
        """
        Initialization method.

        Parameters
        ----------
        input_file : str
            The input job file to read.
        validate_tfrecords : bool
            If True, the corresponding tfrecords files will be created if
            missing.

        """
        self._reader = InputReader(input_file)

        set_float_precision(self._reader['precision'])

        self._dataset = self._get_dataset(validate_tfrecords)
        self._hparams = self._get_hparams()
        self._nn = self._get_nn()
        self._input_file = input_file

    @property
    def nn(self) -> BasicNN:
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
    def hparams(self) -> AttributeDict:
        """
        Return a dict of hyper parameters.
        """
        return self._hparams

    def _backup_input_file(self):
        """
        Copy the input file to `model_dir` for back up.
        """
        dst = join(self._hparams.train.model_dir, basename(self._input_file))
        if realpath(dst) == realpath(self._input_file):
            dst += '.bak'
        if exists(dst):
            os.remove(dst)
        shutil.copyfile(self._input_file, dst)

    def _get_hparams(self):
        """
        Initialize the hyper parameters.
        """
        hparams = AttributeDict(
            seed=self._reader['seed'],
            precision=self._reader['precision'],
            train=AttributeDict(self._reader['train']),
            opt=AttributeDict(self._reader['opt']),
            loss=AttributeDict(self._reader['nn.loss']))

        if not hparams.opt.decay_function:
            hparams.opt.decay_function = None
        if not hparams.train.previous_checkpoint:
            hparams.train.previous_checkpoint = None

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

    def _get_atomic_nn(self, kwargs: dict) -> AtomicNN:
        """
        Initialize an `AtomicNN` or one of its variant.
        """
        hidden_sizes = {}
        for element in kwargs['elements']:
            keypath = f'nn.atomic.layers.{element}'
            value = self._reader[keypath]
            if value is not None:
                hidden_sizes[element] = value

        normalizer = self._reader['nn.atomic.input_normalizer']
        normalization_weights = \
            self._dataset.transformer.get_descriptor_normalization_weights(
                method=normalizer)

        kwargs['hidden_sizes'] = hidden_sizes
        kwargs['normalizer'] = normalizer
        kwargs['normalization_weights'] = normalization_weights

        if self._reader['nn.atomic.arch'] == 'AtomicNN':
            return AtomicNN(**kwargs)
        else:
            kwargs['atomic_static_energy'] = \
                self._dataset.atomic_static_energy
            return AtomicResNN(**kwargs)

    def _get_eam_nn(self, kwargs: dict) -> Union[EamAlloyNN, EamFsNN]:
        """
        Initialize an `EamAlloyNN` or an 'EamFsNN'.
        """

        hidden_sizes = {}
        custom_potentials = {}

        for pot in ('rho', 'embed', 'phi'):
            keypath = f'nn.eam.{pot}'
            if self._reader[keypath] is None:
                continue
            for key in self._reader[keypath].keys():
                value = self._reader[f'nn.eam.{pot}.{key}']
                if value is None:
                    continue
                if isinstance(value, str):
                    if value not in available_potentials:
                        raise ValueError(f"The empirical potential "
                                         f"[{pot}.{value}] is not available")
                    nested_set(custom_potentials, f'{key}.{pot}', value)
                else:
                    nested_set(hidden_sizes, f'{key}.{pot}', value)
                    nested_set(custom_potentials, f'{key}.{pot}', 'nn')

        kwargs.update(dict(hidden_sizes=hidden_sizes,
                           custom_potentials=custom_potentials))

        arch = self._reader['nn.eam.arch']
        if arch == 'EamAlloyNN':
            return EamAlloyNN(**kwargs)
        else:
            return EamFsNN(**kwargs)

    def _get_nn(self):
        """
        Initialize a `BasicNN` using the configs of the input file.
        """
        elements = self._dataset.transformer.elements
        minimize_properties = self._reader['nn.minimize']
        export_properties = self._reader['nn.export']
        activation = self._reader['nn.activation']
        positive_energy_mode = self._reader['nn.positive_energy_mode']
        kwargs = {'elements': elements,
                  'minimize_properties': minimize_properties,
                  'export_properties': export_properties,
                  'activation': activation,
                  'positive_energy_mode': positive_energy_mode}
        if self._reader['dataset.descriptor'] == 'behler':
            nn = self._get_atomic_nn(kwargs)
        else:
            nn = self._get_eam_nn(kwargs)

        # Attach the transformer
        nn.attach_transformer(self._dataset.transformer)
        return nn

    def _get_dataset(self, validate_tfrecords=True):
        """
        Initialize a `Dataset` using the configs of the input file.
        """
        descriptor = self._reader['dataset.descriptor']
        database = connect(self._reader['dataset.sqlite3'])
        name = self._reader['dataset.name']
        rc = self._reader['dataset.rc']
        serial = self._reader['dataset.serial']

        if descriptor == 'behler':
            kwargs = self._reader['nn.atomic.behler']
        else:
            kwargs = {}
        dataset = Dataset(database=database, descriptor=descriptor,
                          name=name, rc=rc, serial=serial, **kwargs)

        test_size = self._reader['dataset.test_size']
        tfrecords_dir = self._reader['dataset.tfrecords_dir']
        if not dataset.load_tfrecords(tfrecords_dir):
            if validate_tfrecords:
                dataset.to_records(
                    tfrecords_dir, test_size=test_size, verbose=True)
            else:
                tf.logging.info("Warning: tfrecords files missing.")
        return dataset

    @staticmethod
    def _check_before_training(hparams: AttributeDict):
        """
        Check the `model_dir` and `previous_checkpoint` before training.
        """
        model_dir = hparams.train.model_dir
        deleted = False

        if exists(model_dir):
            if hparams.train.restart:
                shutil.rmtree(model_dir, ignore_errors=True)
                tf.gfile.MakeDirs(hparams.train.model_dir)
                deleted = True
        else:
            tf.gfile.MakeDirs(hparams.train.model_dir)

        if hparams.train.previous_checkpoint:
            ckpt_dir = dirname(hparams.train.previous_checkpoint)
            if (realpath(ckpt_dir) == realpath(model_dir)) and deleted:
                print(f"Warning: {hparams.train.previous_checkpoint} "
                      f"was already deleted")
                hparams.train.previous_checkpoint = None

    def train_and_evaluate(self, debug=False):
        """
        Initialize a model and train it with `tf.estimator.train_and_evalutate`.
        """

        graph = tf.Graph()

        with graph.as_default():

            dataset = self._dataset
            nn = self._nn
            hparams = self._hparams

            self._check_before_training(hparams)
            self._backup_input_file()

            set_logging_configs(
                logfile=check_path(join(hparams.train.model_dir, 'logfile')))

            tf.logging.info(f'pid={os.getpid()}')
            tf.logging.info(f'seed={self._hparams.seed}')
            tf.logging.info(f'input= \n{str(self._reader)}')

            gpu_options = tf.GPUOptions(allow_growth=True)
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            gpu_options=gpu_options)

            estimator = tf.estimator.Estimator(
                model_fn=nn.model_fn,
                warm_start_from=None,
                model_dir=hparams.train.model_dir,
                config=tf.estimator.RunConfig(
                    save_checkpoints_steps=hparams.train.eval_steps,
                    tf_random_seed=hparams.seed,
                    log_step_count_steps=hparams.train.log_steps,
                    keep_checkpoint_max=hparams.train.max_checkpoints_to_keep,
                    session_config=session_config),
                params=hparams)

            if debug:
                hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline"), ]
            else:
                hooks = None

            train_spec = tf.estimator.TrainSpec(
                input_fn=dataset.input_fn(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    batch_size=hparams.train.batch_size,
                    shuffle=hparams.train.shuffle),
                max_steps=hparams.train.train_steps,
                hooks=hooks)

            eval_spec = tf.estimator.EvalSpec(
                input_fn=dataset.input_fn(
                    mode=tf.estimator.ModeKeys.EVAL,
                    batch_size=hparams.train.eval_batch_size,
                    num_epochs=1,
                    shuffle=False),
                steps=hparams.train.eval_steps,
                start_delay_secs=300,
                # Explicitly set these thresholds to lower values so that every
                # checkpoint can be evaluated.
                throttle_secs=60,
            )
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def export(self, checkpoint=None):
        """
        Export the trained model.
        """
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(
                self._hparams.train.model_dir)

        output_graph_path = join(self._hparams.train.model_dir,
                                 f'{self._dataset.name}.pb')

        self._nn.export(output_graph_path=output_graph_path,
                        checkpoint=checkpoint,
                        keep_tmp_files=False)

        if isinstance(self._nn, (EamAlloyNN, EamFsNN)):
            kwargs = self._reader['nn.eam.setfl']

            if 'lattice' in kwargs:
                lattice = kwargs.pop('lattice')
                lattice_constants = lattice.get('constant', {})
                lattice_types = lattice.get('type', {})
            else:
                lattice_constants = None
                lattice_types = None

            output_setfl = join(self._hparams.train.model_dir,
                                f'{self._dataset.name}.{self._nn.tag}.eam')

            self._nn.export_to_setfl(output_setfl, checkpoint=checkpoint,
                                     lattice_constants=lattice_constants,
                                     lattice_types=lattice_types, **kwargs)


def main(args: Namespace):
    """
    The main function.
    """
    export_latest_only = args.export_latest_only
    validate_tfrecords = not export_latest_only

    manager = TrainingManager(
        args.filename,
        validate_tfrecords=validate_tfrecords)
    if not export_latest_only:
        manager.train_and_evaluate(debug=args.debug)
    manager.export()


def config_parser(parser: ArgumentParser):
    """
    Setup the `ArgumentParser`.
    """
    parser.add_argument(
        'filename',
        type=str,
        help="A cfg file to read."
    )
    parser.add_argument(
        '--export-latest-only',
        action='store_true',
        default=False,
        help="Directly export the lastest checkpoint to a model file and exit."
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Enabled the debugging mode.",
    )
    return parser


if __name__ == "__main__":

    main(config_parser(ArgumentParser()).parse_args().filename)
