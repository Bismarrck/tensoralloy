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

from tensoralloy.dtypes import get_float_dtype
from tensoralloy.utils import GraphKeys
from tensoralloy.test_utils import test_dir
from tensoralloy.nn.utils import log_tensor


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# **Materials Project**
# https://wiki.materialsproject.org/Elasticity_calculations
#
# Note that in this work, conventional unit cells, obtained using
# `pymatgen.symmetry.SpacegroupAnalyzer.get_conventional_standard_structure`
# are employed for all elastic constant calculations. In our experience, these
# cells typically yield more accurate and better converged elastic constants
# than primitive cells, at the cost of more computational time. We suspect this
# has to do with the fact that unit cells often exhibit higher symmetries and
# simpler Brillouin zones than primitive cells (an example is face centered
# cubic cells).
_identifier = "conventional_standard"


# noinspection PyTypeChecker,PyArgumentList
class ElasticConstant(namedtuple('ElasticConstant', ('ijkl', 'value'))):
    """
    This class represents a specifi C_{ijkl}.
    """

    def __new__(cls,
                ijkl: Union[List[int], np.ndarray],
                value: float):
        return super(ElasticConstant, cls).__new__(cls, ijkl, value)


# noinspection PyTypeChecker,PyArgumentList
class Crystal(namedtuple('Crystal', ('name', 'atoms', 'elastic_constants'))):
    """
    A container class for a crystal.
    """

    def __new__(cls,
                name: str,
                atoms: Atoms,
                elastic_constants: List[ElasticConstant]):
        return super(Crystal, cls).__new__(cls, name, atoms, elastic_constants)


built_in_crystals = {
    "Ni": Crystal("Ni", bulk("Ni", cubic=True),
                  [ElasticConstant([0, 0, 0, 0], 276),
                   ElasticConstant([0, 0, 1, 1], 159),
                   ElasticConstant([1, 2, 1, 2], 132)]),
    "Mo": Crystal("Mo", bulk("Mo", cubic=True),
                  [ElasticConstant([0, 0, 0, 0], 472),
                   ElasticConstant([0, 0, 1, 1], 158),
                   ElasticConstant([1, 2, 1, 2], 106)]),
    "Ni4Mo": Crystal("Ni4Mo", read(join(test_dir(),
                                        'crystals',
                                        f'Ni4Mo_mp-11507_{_identifier}.cif')),
                     [ElasticConstant([0, 0, 0, 0], 472),
                      ElasticConstant([0, 0, 1, 1], 158),
                      ElasticConstant([1, 2, 1, 2], 106)]),
    "Ni3Mo": Crystal("Ni3Mo", read(join(test_dir(),
                                        'crystals',
                                        f'Ni3Mo_mp-11506_{_identifier}.cif')),
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


def get_elastic_constant_loss(nn,
                              list_of_crystal: List[Union[Crystal, str]],
                              weight=1.0,
                              constraint_weight=1.0):
    """
    Return a special loss: RMSE (GPa) of certain elastic constants of
    certain bulk solids.

    Parameters
    ----------
    nn : BasicNN
        A `BasicNN`. Its weights will be reused.
    list_of_crystal : List[Crystal] or List[str]
        A list of `Crystal` objects. It can also be a list of str as the names
        of the built-in crystals.
    weight : float
        The weight of the loss contributed by elastic constants.
    constraint_weight : float
        The weight of the loss contributed by the constraints.

    """
    configs = nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['energy', 'forces', 'stress']

    with tf.name_scope("Elastic/"):

        for crystal in list_of_crystal:
            if isinstance(crystal, str):
                crystal = built_in_crystals[crystal]
            elif not isinstance(crystal, Crystal):
                raise ValueError(
                    "`crystal` must be a str or a `Crystal` object!")

            symbols = set(crystal.atoms.get_chemical_symbols())
            for symbol in symbols:
                if symbol not in nn.elements:
                    raise ValueError(f"{symbol} is not supported!")

            predictions = []
            labels = []
            constraints = []

            with tf.name_scope(f"{crystal.name}"):
                elastic_nn = nn.__class__(**configs)
                elastic_clf = nn.transformer.as_descriptor_transformer()
                elastic_nn.attach_transformer(elastic_clf)

                features = elastic_clf.get_constant_features(crystal.atoms)

                output = elastic_nn.build(
                    features=features,
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=True)

                stress = tf.get_default_graph().get_tensor_by_name(
                    f'Elastic/{crystal.name}/Output/Stress/Full/stress:0')

                with tf.name_scope("Constraints"):
                    constraints.extend([
                        tf.linalg.norm(output.forces, name='forces')])

                with tf.name_scope("Cijkl"):
                    for elastic_constant in crystal.elastic_constants:
                        i, j, k, l = elastic_constant.ijkl
                        dsdhij = tf.identity(tf.gradients(
                            stress[i, j],
                            features.cells)[0], name=f'dsdh{i}{j}')
                        cij = tf.matmul(
                            tf.transpose(dsdhij),
                            tf.identity(features.cells),
                            name=f'h_dsdh{i}{j}')
                        cij = tf.math.truediv(cij, features.volume)
                        cij = tf.math.truediv(cij, GPa, name=f'c{i}{j}')
                        vi = voigt_notation(i, j)
                        vj = voigt_notation(k, l)
                        cijkl = tf.identity(cij[k, l], name=f'C{vi}{vj}')

                        log_tensor(cijkl)

                        predictions.append(cijkl)
                        labels.append(
                            tf.convert_to_tensor(elastic_constant.value,
                                                 dtype=stress.dtype,
                                                 name=f'refC{vi}{vj}'))

                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, cijkl)

        with tf.name_scope("Loss"):

            # Loss contribution from elastic constants
            with tf.name_scope("Elastic"):
                predictions = tf.stack(predictions, name='predictions')
                labels = tf.stack(labels, name='labels')

                mse = tf.reduce_mean(tf.squared_difference(predictions, labels),
                                     name='mse')
                dtype = get_float_dtype()
                eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
                mse = tf.add(mse, eps, name='mse/safe')
                weight = tf.convert_to_tensor(weight, dtype, name='weight')
                raw_loss = tf.sqrt(mse, name='rmse')
                e_loss = tf.multiply(raw_loss, weight, name='weighted/loss')

            # Loss contribution from constraints because the 2-norm of the total
            # forces and stress of the crystal structure should be zero.
            with tf.name_scope("Constraint"):
                c_loss = tf.add_n(constraints, name='loss')
                weight = tf.convert_to_tensor(
                    constraint_weight, dtype, name='weight')
                c_loss = tf.multiply(c_loss, weight, name='weighted/loss')

        total_loss = tf.add(e_loss, c_loss, name='total_loss')

        tf.add_to_collection(GraphKeys.TRAIN_METRICS, e_loss)
        tf.add_to_collection(GraphKeys.TRAIN_METRICS, c_loss)

        return total_loss
