#!coding=utf-8
"""
This module can be used to infer RMSE loss of elastic constants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from tensorflow_estimator import estimator as tf_estimator
from ase.units import GPa
from typing import List, Union
from itertools import product

from tensoralloy.nn.constraint.data import Crystal, get_crystal
from tensoralloy.nn.constraint.voigt import voigt_notation
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.dataclasses import ElasticConstraintOptions
from tensoralloy.precision import get_float_dtype
from tensoralloy.utils import GraphKeys


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_cijkl_op(total_stress: tf.Tensor, cell: tf.Tensor, volume: tf.Tensor,
                  i: int, j: int, k: int, l: int, name=None):
    """
    Return the Op to compute C_{ijkl}.
    """
    dsdhij = tf.identity(tf.gradients(
        total_stress[i, j],
        cell)[0], name=f'dsdh{i}{j}')
    cij = tf.matmul(
        tf.transpose(dsdhij),
        tf.identity(cell),
        name=f'h_dsdh{i}{j}')
    cij = tf.math.truediv(cij, volume)
    cij = tf.math.truediv(
        cij, tf.convert_to_tensor(GPa, dtype=cell.dtype, name='GPa'),
        name=f'c{i}{j}')
    if name is None:
        name = f'c{i}{j}{k}{l}'
    return tf.identity(cij[k, l], name=name)


def get_elastic_constat_tensor_op(total_stress: tf.Tensor, cell: tf.Tensor,
                                  volume: tf.Tensor, name='elastic',
                                  verbose=False):
    """
    Return the Op to compute the full elastic constant tensor.

    Parameters
    ----------
    total_stress : tf.Tensor
        The total stress (with unit 'eV') tensor. The shape should be `[3, 3]`.
    cell : tf.Tensor
        The 3x3 lattice tensor. The shape should be `[3, 3]`.
    volume : tf.Tensor
        The lattice volume tensor.
    name : str
        The name of the output tensor.
    verbose : bool
        A boolean flag.

    Returns
    -------
    cijkl : tf.Tensor
        The full elastic constant tensor of shape `[6, 6]` with unit GPa.

    """
    with tf.name_scope("Cijkl"):
        assert total_stress.shape.as_list() == [3, 3]
        assert cell.shape.as_list() == [3, 3]
        components = []
        v2c_map = []
        filled = np.zeros((6, 6), dtype=bool)
        for (i, j, k, l) in product([0, 1, 2], repeat=4):
            vi = voigt_notation(i, j, return_py_index=True)
            vj = voigt_notation(k, l, return_py_index=True)
            if filled[vi, vj]:
                continue
            cijkl = _get_cijkl_op(total_stress, cell, volume, i, j, k, l)
            components.append(cijkl)
            v2c_map.append((vi, vj))
            filled[vi, vj] = True
        vector = tf.stack(components)
        elastic = tf.scatter_nd(v2c_map, vector, shape=(6, 6), name=name)
        if verbose:
            log_tensor(elastic)
        return elastic


def get_elastic_constant_loss(nn,
                              list_of_crystal: List[Union[Crystal, str]],
                              options: ElasticConstraintOptions = None,
                              weight=1.0):
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
    options : ElasticConstraintOptions
        The options of the loss contributed by the constraints.
    weight : float
        The weight of the loss contributed by elastic constants.

    """
    if options is None:
        options = ElasticConstraintOptions()

    configs = nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['energy', 'forces', 'stress']

    with tf.name_scope("Elastic/"):

        predictions = []
        labels = []
        cijkl_weights = []
        constraints = {'forces': [], 'stress': []}

        for crystal_or_name_or_file in list_of_crystal:
            crystal = get_crystal(crystal_or_name_or_file)

            symbols = set(crystal.atoms.get_chemical_symbols())
            for symbol in symbols:
                if symbol not in nn.elements:
                    raise ValueError(f"{symbol} is not supported!")

            with tf.name_scope(f"{crystal.name}/{crystal.phase}"):
                elastic_nn = nn.__class__(**configs)
                elastic_clf = nn.transformer.as_descriptor_transformer()
                elastic_nn.attach_transformer(elastic_clf)

                features = elastic_clf.get_constant_features(crystal.atoms)

                output = elastic_nn.build(
                    features=features,
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=True)
                total_stress = output.total_stress
                cell = features.cells
                volume = features.volume

                with tf.name_scope("Constraints"):
                    constraints['forces'].append(
                        tf.linalg.norm(output.forces, name='forces'))

                    if options.use_kbar:
                        unit = tf.constant(10.0 / GPa, dtype=total_stress.dtype,
                                           name='unit')
                        value = tf.identity(
                            tf.linalg.norm(
                                tf.math.multiply(output.stress, unit)),
                            name='kbar')
                    else:
                        unit = tf.constant(1e4 / GPa, dtype=total_stress.dtype,
                                           name='unit')
                        value = tf.identity(
                            tf.linalg.norm(
                                tf.math.multiply(output.stress, unit)),
                            name='bar')
                    constraints['stress'].append(value)
                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, value)
                    tf.add_to_collection(GraphKeys.EVAL_METRICS, value)

                with tf.name_scope("Cijkl"):
                    for elastic_constant in crystal.elastic_constants:
                        i, j, k, l = elastic_constant.ijkl
                        vi = voigt_notation(i, j)
                        vj = voigt_notation(k, l)
                        cijkl = _get_cijkl_op(
                            total_stress, cell, volume, i, j, k, l,
                            name=f"C{vi}{vj}")
                        log_tensor(cijkl)
                        predictions.append(cijkl)
                        labels.append(
                            tf.convert_to_tensor(elastic_constant.value,
                                                 dtype=total_stress.dtype,
                                                 name=f'C{vi}{vj}/ref'))
                        cijkl_weights.append(
                            tf.convert_to_tensor(elastic_constant.weight,
                                                 dtype=total_stress.dtype,
                                                 name=f'C{vi}{vj}/weight'))

                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, cijkl)
                        tf.add_to_collection(GraphKeys.EVAL_METRICS, cijkl)

        with tf.name_scope("Loss"):

            # Loss contribution from elastic constants
            with tf.name_scope("Elastic"):
                predictions = tf.stack(predictions, name='predictions')
                labels = tf.stack(labels, name='labels')
                cijkl_weights = tf.stack(cijkl_weights, name='weights')

                sd = tf.multiply(cijkl_weights,
                                 tf.squared_difference(predictions, labels),
                                 name='sd/weighted')
                mse = tf.reduce_mean(sd, name='mse')
                mae = tf.reduce_mean(tf.abs(predictions - labels), name='mae')
                dtype = get_float_dtype()
                eps = tf.constant(dtype.eps, dtype=dtype, name='eps')
                mse = tf.add(mse, eps, name='mse/safe')
                weight = tf.convert_to_tensor(weight, dtype, name='weight')
                raw_loss = tf.sqrt(mse, name='rmse')
                e_loss = tf.multiply(raw_loss, weight, name='weighted/loss')

            # Loss contribution from constraints because the 2-norm of the total
            # forces and stress of the crystal structure should be zero.
            with tf.name_scope("Constraint"):
                f_loss = tf.add_n(constraints['forces'], name='f_loss')
                f_weight = tf.convert_to_tensor(
                    options.forces_weight, dtype, name='weight/f')
                f_loss = tf.multiply(f_loss, f_weight, name='weighted/f_loss')

                p_loss = tf.add_n(constraints['stress'], name='p_loss')
                p_weight = tf.convert_to_tensor(
                    options.stress_weight, dtype, name='weight/p')
                p_loss = tf.multiply(p_loss, p_weight, name='weighted/p_loss')

                c_loss = tf.add(p_loss, f_loss, name='c_loss')

        total_loss = tf.add(e_loss, c_loss, name='total_loss')

        tf.add_to_collection(GraphKeys.TRAIN_METRICS, e_loss)
        tf.add_to_collection(GraphKeys.TRAIN_METRICS, c_loss)
        tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)

        return total_loss