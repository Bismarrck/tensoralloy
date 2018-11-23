# coding=utf-8
"""
This module defines the `ConfigParser` for this project.
"""
from __future__ import print_function, absolute_import

import configparser
from ase.db import connect
from os.path import splitext, basename, join, exists, dirname
from typing import Callable
from shutil import rmtree

from misc import Defaults, AttributeDict, safe_select
from dataset import Dataset
from nn import AtomicNN, AtomicResNN, EamNN, BasicNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


_UNSET = object()


def _parse_list(dtype: Callable, x: str):
    """
    A helper function returning a list of `dtype` objects from a comma-seprated
    string.
    """
    if x is None or x.strip() == '':
        return None
    return [dtype(e.strip()) for e in x.strip().split(',')]


class ConfigParser(configparser.ConfigParser):
    """
    A specific implementation of `configparser.ConfigParser`.
    """

    def __init__(self, filename):
        """
        Initialization method.

        Parameters
        ----------
        filename : str
            The config file to read.

        """
        super(ConfigParser, self).__init__(
            converters={
                'floats': lambda x: _parse_list(float, x),
                'ints': lambda x: _parse_list(int, x)},
            interpolation=configparser.ExtendedInterpolation())
        with open(filename) as fp:
            self._read(fp, filename)
        self._dataset = self._read_dataset()
        self._nn = self._build_nn()
        self._hparams = self._read_hyperparams()

    @property
    def dataset(self):
        """
        Return the `Dataset` intialized from this config.
        """
        return self._dataset

    @property
    def nn(self):
        """
        Return an instance of `BasicNN` initialized from this config.
        """
        return self._nn

    @property
    def hyper_params(self):
        """
        Return a dict of hyper parameters.
        """
        return self._hparams

    def _get_conv(self, section, option, conv, *, raw=False, vars=None,
                  fallback=_UNSET, **kwargs):
        """
        Always return `fallback` if provided.
        """
        try:
            return self._get(section, conv, option, raw=raw, vars=vars,
                             **kwargs)
        except Exception as excp:
            if fallback is _UNSET:
                raise excp
            return fallback

    def _read_dataset(self):
        """
        Initialize a `Dataset`.
        """
        section = 'dataset'
        filename = self[section]['sqlite3']
        descriptor = self[section]['descriptor']

        if descriptor not in ('eam', 'behler'):
            raise NotImplementedError(
                f"The descriptor `{descriptor}` is not implemented!")

        database = connect(filename)
        name = self[section].get('name', splitext(basename(filename))[0])
        rc = self[section].getfloat('rc', Defaults.rc)
        k_max = self[section].getint('k_max', Defaults.k_max)

        if descriptor == 'behler':
            section = 'behler'
            eta = self[section].getfloats('eta', Defaults.eta)
            beta = self[section].getfloats('beta', Defaults.beta)
            gamma = self[section].getfloats('gamma', Defaults.gamma)
            zeta = self[section].getfloats('zeta', Defaults.zeta)
            kwargs = {'eta': eta, 'beta': beta, 'gamma': gamma, 'zeta': zeta}
        else:
            kwargs = {}

        dataset = Dataset(database=database, descriptor=descriptor, name=name,
                          k_max=k_max, rc=rc, **kwargs)

        section = 'tfrecords'
        test_size = self[section].getint('test_size', 1000)
        tfrecords_dir = safe_select(
            self[section].get('tfrecords_dir', '.'), '.')
        if not dataset.load_tfrecords(tfrecords_dir):
            dataset.to_records(tfrecords_dir, test_size=test_size, verbose=True)
        return dataset

    def _get_nn_common_args(self):
        """
        Return the common args for building a `BasicNN`.
        """
        section = 'nn'

        l2_weight = self[section].getfloat('l2_weight', 0.0)
        activation = safe_select(
            self[section].get('activation', 'leaky_relu'), 'leaky_relu')

        forces = safe_select(self[section].getboolean('forces', True), True)
        forces = self._dataset.forces and forces

        stress = safe_select(self[section].getboolean('stress', False), False)
        stress = self._dataset.stress and stress

        total_pressure = safe_select(
            self[section].getboolean('total_pressure', False), False)
        total_pressure = self._dataset.stress and total_pressure

        hidden_sizes = {}
        for element in self._dataset.transformer.elements:
            hidden_sizes[element] = self[section].getints(
                element, Defaults.hidden_sizes)

        return {'elements': self._dataset.transformer.elements,
                'hidden_sizes': hidden_sizes,
                'activation': activation, 'forces': forces, 'stress': stress,
                'total_pressure': total_pressure, 'l2_weight': l2_weight}

    def _build_nn(self) -> BasicNN:
        """
        Build an `AtomicNN` with the given config and dataset.
        """
        section = 'nn'

        args = self._get_nn_common_args()
        arch = safe_select(self[section].get('arch', 'AtomicNN'), 'AtomicNN')

        if self._dataset.descriptor == 'eam':
            if arch != 'EamNN':
                raise ValueError(f"The arch {arch} is not supported "
                                 f"for '{self._dataset.descriptor}'.")
            return EamNN(**args)

        else:
            normalizer = safe_select(
                self[section].get('input_normalizer', None), None)
            normalization_weights = \
                self._dataset._transformer.get_descriptor_normalization_weights(
                    method=normalizer)
            args.update({'normalizer': normalizer,
                         'normalization_weights': normalization_weights})
            if arch == 'AtomicNN':
                nn = AtomicNN(**args)
            elif arch == 'AtomicResNN':
                args['atomic_static_energy'] = \
                    self._dataset.atomic_static_energy
                nn = AtomicResNN(**args)
            else:
                raise ValueError(f"The arch {arch} is not supported "
                                 f"for '{self._dataset.descriptor}'.")
            return nn

    def _read_hyperparams(self):
        """
        Read hyper parameters.
        """
        section = 'opt'

        method = safe_select(self[section].get('optimizer', 'adam'), 'adam')
        learning_rate = self[section].getfloat(
            'learning_rate', Defaults.learning_rate)

        decay_function = self[section].get('decay_function', None)
        decay_function = safe_select(decay_function, None)

        decay_rate = self[section].getfloat('decay_rate', None)
        decay_steps = self[section].getint('decay_steps', None)
        staircase = self[section].getboolean('staircase', False)

        opt = AttributeDict(method=method,
                            learning_rate=learning_rate,
                            decay_function=decay_function,
                            decay_steps=decay_steps,
                            decay_rate=decay_rate,
                            staircase=staircase)

        section = 'train'
        model_dir = self[section].get('model_dir')
        batch_size = self[section].getint('batch_size', 50)
        shuffle = safe_select(self[section].getboolean('shuffle', True), True)

        if self._dataset.test_size % batch_size != 0:
            eval_batch_size = next(x for x in range(batch_size, 0, -1)
                                   if self._dataset.test_size % x == 0)
            print("Warning: batch_size is reduced to {:d} for eval".format(
                eval_batch_size, self._dataset.test_size, batch_size))
        else:
            eval_batch_size = batch_size

        train_steps = self[section].getint('train_steps', 10000)
        eval_steps = self[section].getint('eval_steps', 1000)
        eval_dir = join(model_dir, 'eval')
        summary_steps = self[section].getint('summary_steps', 100)
        log_steps = self[section].getint('log_steps', 100)
        max_checkpoints_to_keep = self[section].getint(
            'max_checkpoints_to_keep', 10)
        profile_steps = self[section].getint('profile_steps', 0)

        restart = self[section].getboolean('restart', True)
        removed = []
        if not restart:
            if exists(model_dir):
                rmtree(model_dir)
                removed.append(model_dir)
            if exists(eval_dir):
                rmtree(eval_dir)
                removed.append(eval_dir)

        previous_checkpoint = safe_select(
            self[section].get('previous_checkpoint', None), None)
        if restart and previous_checkpoint:
            if dirname(previous_checkpoint) in removed:
                print("Warning: {} was already deleted!".format(
                    previous_checkpoint))
            previous_checkpoint = None

        train = AttributeDict(model_dir=model_dir,
                              previous_checkpoint=previous_checkpoint,
                              shuffle=shuffle,
                              batch_size=batch_size,
                              train_steps=train_steps,
                              eval_steps=eval_steps,
                              eval_dir=eval_dir,
                              eval_batch_size=eval_batch_size,
                              summary_steps=summary_steps,
                              log_steps=log_steps,
                              max_checkpoints_to_keep=max_checkpoints_to_keep,
                              profile_steps=profile_steps)

        return AttributeDict(opt=opt, train=train)
