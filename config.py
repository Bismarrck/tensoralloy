# coding=utf-8
"""
This module defines the `ConfigParser` for this project.
"""
from __future__ import print_function, absolute_import

import configparser
from ase.db import connect
from os.path import splitext, basename, join, exists
from typing import Callable
from shutil import rmtree

from misc import Defaults, AttributeDict, safe_select
from dataset import Dataset
from nn import AtomicNN, AtomicResNN

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
        Return the `AtomicNN` initialized from this config.
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

        if descriptor != 'behler':
            raise NotImplementedError(
                "Only Behler's Symmetry Function is implemented at this time!")

        database = connect(filename)
        name = self[section].get('name', splitext(basename(filename))[0])
        rc = self[section].getfloat('rc', Defaults.rc)
        k_max = self[section].getint('k_max', Defaults.k_max)
        zeros_forces_mode = self[section].getboolean('zero_forces_mode', False)

        section = 'behler'
        eta = self[section].getfloats('eta', Defaults.eta)
        beta = self[section].getfloats('beta', Defaults.beta)
        gamma = self[section].getfloats('gamma', Defaults.gamma)
        zeta = self[section].getfloats('zeta', Defaults.zeta)

        dataset = Dataset(database=database, name=name, k_max=k_max, rc=rc,
                          eta=eta, beta=beta, gamma=gamma, zeta=zeta)
        if zeros_forces_mode:
            dataset.use_zero_forces()

        section = 'tfrecords'
        test_size = self[section].getint('test_size', 1000)
        tfrecords_dir = self[section].get('tfrecords_dir', '.')
        if not dataset.load_tfrecords(tfrecords_dir):
            dataset.to_records(tfrecords_dir, test_size=test_size, verbose=True)
        return dataset

    def _build_nn(self) -> AtomicNN:
        """
        Build an `AtomicNN` with the given config and dataset.
        """
        section = 'nn'

        l2_weight = self[section].getfloat('l2_weight', 0.0)
        activation = self[section].get('activation', 'leaky_relu')
        forces = self._dataset.forces

        transformer = self._dataset.transformer
        elements = transformer.elements
        hidden_sizes = {}
        for element in elements:
            hidden_sizes[element] = self[section].getints(
                element, Defaults.hidden_sizes)

        normalizer = self[section].get('input_normalizer', None)
        normalizer_weights = transformer.get_descriptor_normalization_weights(
            method=normalizer)
        arch = self[section].get('arch', 'AtomicNN')

        if arch == 'AtomicNN':
            nn = AtomicNN(elements=elements, hidden_sizes=hidden_sizes,
                          activation=activation, l2_weight=l2_weight,
                          forces=forces, normalizer=normalizer,
                          normalization_weights=normalizer_weights)
        elif arch == 'AtomicResNN':
            atomic_static_energy = self._dataset.atomic_static_energy
            nn = AtomicResNN(elements=elements, hidden_sizes=hidden_sizes,
                             activation=activation, l2_weight=l2_weight,
                             forces=forces, normalizer=normalizer,
                             normalization_weights=normalizer_weights,
                             atomic_static_energy=atomic_static_energy)
        else:
            raise ValueError(f"The arch {arch} is not supported. Please use: "
                             f"AtomicNN or AtomicResNN.")
        return nn

    def _read_hyperparams(self):
        """
        Read hyper parameters.
        """
        section = 'opt'

        method = self[section].get('optimizer', 'adam')
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
        if not restart:
            if exists(model_dir):
                rmtree(model_dir)
            if exists(eval_dir):
                rmtree(eval_dir)

        train = AttributeDict(model_dir=model_dir,
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
