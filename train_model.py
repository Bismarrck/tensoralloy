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
from typing import Callable, List

from dataset import Dataset
from misc import safe_select


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _parse_list(dtype: Callable, list_str: str):
    """
    A helper function returning a list of `dtype` objects from a comma-seprated
    string.
    """
    if list_str is None:
        return None
    return [dtype(e.strip()) for e in list_str.split(',')]


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


def train_and_evaluate(config: ConfigParser):

    dataset = get_dataset_from_config(config)



def main(cfgfile):
    """
    The main function.
    """
    config = ConfigParser()
    config.read(cfgfile)
    train_and_evaluate(config)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        'filename',
        type=str,
        help="A cfg file to read."
    )
    main(parser.parse_args().filename)
