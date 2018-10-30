# coding=utf-8
"""
This module is used to train the model
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from ase.db import connect
from os.path import splitext, basename
from argparse import ArgumentParser
from configparser import ConfigParser
from typing import Callable
from functools import partial

from dataset import Dataset, TrainableProperty
from nn import AtomicNN, get_activation_fn, get_optimizer, get_learning_rate
from misc import Defaults


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _parse_list(dtype: Callable, list_str: str):
    """
    A helper function returning a list of `dtype` objects from a comma-seprated
    string.
    """
    if list_str is None or list_str.strip() == '':
        return None
    return [dtype(e.strip()) for e in list_str.strip().split(',')]


def get_dataset_from_config(config: ConfigParser) -> Dataset:
    """
    Initialize a `Dataset` with the given config.
    """
    section = 'dataset'
    filename = config.get(section, 'sqlite3')
    descriptor = config.get(section, 'descriptor')

    if descriptor != 'behler':
        raise NotImplementedError(
            "Only Behler's Symmetry Function is implemented at this time!")

    database = connect(filename)
    name = config.get(section, 'name', fallback=splitext(basename(filename))[0])
    rc = config.getfloat(section, 'rc', fallback=None)
    k_max = config.getint(section, 'k_max', fallback=2)

    section = 'behler'
    eta = _parse_list(float, config.get(section, 'eta', fallback=None))
    beta = _parse_list(float, config.get(section, 'beta', fallback=None))
    gamma = _parse_list(float, config.get(section, 'gamma', fallback=None))
    zeta = _parse_list(float, config.get(section, 'zeta', fallback=None))

    dataset = Dataset(database=database, name=name, k_max=k_max, rc=rc, eta=eta,
                      beta=beta, gamma=gamma, zeta=zeta)

    section = 'tfrecords'
    test_size = config.getint(section, 'test_size', fallback=1000)
    tfrecords_dir = config.get(section, 'tfrecords_dir', fallback='.')
    parallel = config.getboolean(section, 'build_in_parallel', fallback=True)
    if not dataset.load_tfrecords(tfrecords_dir):
        dataset.to_records(
            tfrecords_dir, test_size=test_size, parallel=parallel)
    return dataset


def build_nn(dataset: Dataset, config: ConfigParser) -> AtomicNN:
    """
    Build an `AtomicNN` with the given config and dataset.
    """
    section = 'nn'

    l2_weight = config.getfloat(section, 'l2_weight', fallback=0.0)
    fn_name = config.get(section, 'activation_fn', fallback='leaky_relu')
    activation_fn = get_activation_fn(fn_name)
    forces = TrainableProperty.forces in dataset.trainable_properties

    elements = dataset.descriptor.elements
    hidden_sizes = {}
    for element in elements:
        if element in config.options(section):
            hidden_sizes[element] = _parse_list(int,
                                                config.get(section, element))
        else:
            hidden_sizes[element] = Defaults.hidden_sizes

    nn = AtomicNN(elements=dataset.descriptor.elements,
                  hidden_sizes=hidden_sizes,
                  activation_fn=activation_fn,
                  l2_weight=l2_weight,
                  forces=forces)
    return nn


def get_model_fn(nn: AtomicNN, config: ConfigParser) -> Callable:
    """
    Construct an optimizer initializer function and return the model function.
    """
    section = 'optimizer'

    method = config.get(section, 'optimizer', fallback='adam')
    learning_rate = config.getfloat(section, 'learning_rate',
                                    fallback=Defaults.learning_rate)
    decay_function = config.get(section, 'decay_function', fallback=None)
    if decay_function is None or decay_function == '':
        learning_rate = get_learning_rate(None, learning_rate)
        optimizer_initializer = lambda _: get_optimizer(
            learning_rate, method=method)

    else:
        decay_rate = config.getfloat(section, 'decay_rate', fallback=None)
        decay_steps = config.getint(section, 'decay_steps', fallback=None)
        staircase = config.getboolean(section, 'staircase', fallback=False)
        learning_rate_fn = partial(get_learning_rate,
                                   learning_rate=learning_rate,
                                   decay_function=decay_function,
                                   decay_steps=decay_steps,
                                   decay_rate=decay_rate,
                                   staircase=staircase)
        optimizer_initializer = lambda global_step: get_optimizer(
            learning_rate_fn(global_step), method=method)

    return nn.model_fn(optimizer_initializer=optimizer_initializer)


def train_and_evaluate(config: ConfigParser):
    """
    Train and evaluate with the given configuration.

    This method is built upon `tr.estimator.train_and_evalutate`.

    """
    dataset = get_dataset_from_config(config)
    nn = build_nn(dataset, config)
    model_fn = get_model_fn(nn, config)

    section = 'train'
    model_dir = config.get(section, 'model_dir')
    batch_size = config.getint(section, 'batch_size', fallback=50)
    train_steps = config.getint(section, 'train_steps', fallback=10000)
    eval_steps = config.getint(section, 'eval_steps', fallback=1000)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn=dataset.input_fn(batch_size=batch_size, shuffle=True),
        max_steps=train_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=dataset.input_fn(batch_size=batch_size, shuffle=False),
        steps=eval_steps,
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(filename):
    """
    The main function.
    """
    config = ConfigParser()
    config.read(filename)
    train_and_evaluate(config)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        'filename',
        type=str,
        help="A cfg file to read."
    )
    main(parser.parse_args().filename)
