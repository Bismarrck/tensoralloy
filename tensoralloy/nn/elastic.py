#!coding=utf-8
"""
This module can be used to infer RMSE loss of elastic constants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from tensorflow_estimator import estimator as tf_estimator
from ase.units import GPa
from ase import Atoms
from ase.build import bulk
from ase.io import read
from os.path import join
from collections import namedtuple
from typing import List, Union

from tensoralloy.nn.basic import BasicNN
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.dtypes import get_float_dtype
from tensoralloy.utils import GraphKeys
from tensoralloy.test_utils import test_dir


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# noinspection PyTypeChecker,PyArgumentList
class ElasticConstant(namedtuple('ElasticConstant', ('ijkl', 'value'))):

    def __new__(cls,
                ijkl: Union[List[int], np.ndarray],
                value: float):
        return super(ElasticConstant, cls).__new__(cls, ijkl, value)


# noinspection PyTypeChecker,PyArgumentList
class Crystal(namedtuple('Crystal', ('name', 'atoms', 'elastic_constants'))):

    def __new__(cls,
                name: str,
                atoms: Atoms,
                elastic_constants: List[ElasticConstant]):
        return super(Crystal, cls).__new__(cls, atoms, elastic_constants)


_built_in_crystals = {
    "Ni": Crystal("Ni", bulk("Ni"),
                  [ElasticConstant([0, 0, 0, 0], 276),
                   ElasticConstant([0, 0, 1, 1], 159),
                   ElasticConstant([1, 2, 1, 2], 132)]),
    "Mo": Crystal("Mo", bulk("Mo"),
                  [ElasticConstant([0, 0, 0, 0], 472),
                   ElasticConstant([0, 0, 1, 1], 158),
                   ElasticConstant([1, 2, 1, 2], 106)]),
    "Ni4Mo": Crystal("Ni4Mo", read(join(test_dir(),
                                        'crystals',
                                        'Ni4Mo_mp-11507_primitive.cif')),
                    [ElasticConstant([0, 0, 0, 0], 472),
                     ElasticConstant([0, 0, 1, 1], 158),
                     ElasticConstant([1, 2, 1, 2], 106)]),
    "Ni3Mo": Crystal("Ni3Mo", read(join(test_dir(),
                                        'crystals',
                                        'Ni3Mo_mp-11506_primitive.cif')),
                     [ElasticConstant([0, 0, 0, 0], 385),
                      ElasticConstant([0, 0, 0, 1], 166),
                      ElasticConstant([0, 0, 0, 2], 145),
                      ElasticConstant([1, 1, 2, 2], 131),
                      ElasticConstant([1, 1, 1, 1], 402),
                      ElasticConstant([2, 2, 2, 2], 402),
                      ElasticConstant([1, 2, 1, 2], 94)]),
}


def voigt_notation(i, j):
    """
    Return the Voigt notation given two indices (start from zero).
    """
    if i == j:
        return i + 1
    elif (i == 1 and j == 2) or (i == 2 and j == 1):
        return 4
    elif (i == 0 and j == 2) or (i == 2 and j == 0):
        return 5
    else:
        return 6


def get_elastic_constant_loss(nn: BasicNN, list_of_crystal: List[Crystal]):
    """
    Return a special loss: RMSE (GPa) of certain elastic constants of
    certain bulk solids.

    Parameters
    ----------
    nn : BasicNN

    list_of_crystal : List[Crystal]

    """
    configs = nn.as_dict()
    configs.pop('class')
    clf = nn.transformer
    assert isinstance(clf, BatchDescriptorTransformer)

    with tf.name_scope("Elastic/"):

        for crystal in list_of_crystal:

            symbols = set(crystal.atoms.get_chemical_symbols())
            for symbol in symbols:
                if symbol not in nn.elements:
                    raise ValueError(f"{symbol} is not supported!")

            predictions = []
            labels = []

            with tf.name_scope(f"{crystal.name}"):
                elastic_nn = nn.__class__(**configs)
                elastic_clf = clf.as_descriptor_transformer()
                elastic_nn.attach_transformer(elastic_clf)
                elastic_nn.build(
                    features=elastic_clf.get_constant_feed(crystal.atoms),
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=False)

                right = tf.get_default_graph().get_tensor_by_name(
                    f'Elastic/{crystal.name}/Output/Stress/Full/Right/right:0')

                with tf.name_scope("Cijkl"):
                    for elastic_constant in crystal.elastic_constants:
                        i, j, k, l = elastic_constant.ijkl
                        dsdhij = tf.identity(tf.gradients(
                            right[i, j],
                            elastic_clf.placeholders.cells)[0],
                                             name=f'dsdh{i}{j}')
                        cij = tf.matmul(
                            tf.transpose(dsdhij),
                            tf.identity(elastic_clf.placeholders.cells),
                            name=f'h_dsdh{i}{j}')
                        cij = tf.div(cij, elastic_clf.placeholders.volume)
                        cij = tf.div(cij, GPa, name=f'c{i}{j}')
                        vi = voigt_notation(i, j)
                        vj = voigt_notation(k, l)
                        cijkl = tf.gather_nd(cij, [k, l], name=f'C{vi}{vj}')
                        predictions.append(cijkl)
                        labels.append(
                            tf.convert_to_tensor(elastic_constant.value,
                                                 dtype=right.dtype,
                                                 name=f'refC{vi}{vj}'))

                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, cijkl)

        predictions = tf.stack(predictions, name='predictions')
        labels = tf.stack(labels, name='labels')

        mse = tf.reduce_mean(tf.squared_difference(predictions, labels),
                             name='mse')
        mae = tf.reduce_mean(tf.abs(predictions - labels), name='mae')
        dtype = get_float_dtype()
        eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
        mse = tf.add(mse, eps, name='mse/safe')
        loss = tf.sqrt(mse, name='rmse')

        tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)
        tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)

        return loss
