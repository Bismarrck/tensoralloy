#!coding=utf-8
"""
This module can be used to infer RMSE loss of elastic constants.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator
from ase.units import GPa
from typing import List, Union, Tuple

from tensoralloy.nn.constraint.data import Crystal, get_crystal
from tensoralloy.nn.constraint.voigt import voigt_notation, voigt_to_ij
from tensoralloy.nn.utils import log_tensor, is_first_replica
from tensoralloy.nn.dataclasses import ElasticConstraintOptions
from tensoralloy.precision import get_float_dtype
from tensoralloy.utils import GraphKeys


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_cijkl_op(total_stress: tf.Tensor, cell: tf.Tensor, volume: tf.Tensor,
                  i: int, j: int, kl: List[Tuple[int, int]]):
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
        name=f'c{i}{j}xx/GPa')
    groups = {}
    for (k, l) in kl:
        name = f'c{i}{j}{k}{l}'
        groups[(k, l)] = tf.identity(cij[k, l], name=name)
    return groups


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
        for vi in range(6):
            i, j = voigt_to_ij(vi, is_py_index=True)
            pairs = []
            for vj in range(6):
                k, l = voigt_to_ij(vj, is_py_index=True)
                pairs.append((k, l))
            ops = _get_cijkl_op(total_stress, cell, volume, i, j, pairs)
            for (k, l), cijkl in ops.items():
                vj = voigt_notation(k, l, return_py_index=True)
                components.append(cijkl)
                v2c_map.append((vi, vj))
        vector = tf.stack(components)
        elastic = tf.scatter_nd(v2c_map, vector, shape=(6, 6), name=name)
        if verbose:
            log_tensor(elastic)
        return elastic


def get_elastic_constant_loss(base_nn,
                              list_of_crystal: List[Union[Crystal, str]],
                              options: ElasticConstraintOptions = None,
                              weight=1.0,
                              verbose=True):
    """
    Return a special loss: RMSE (GPa) of certain elastic constants of
    certain bulk solids.

    Parameters
    ----------
    base_nn : BasicNN
        A `BasicNN`. Its weights will be reused.
    list_of_crystal : List[Crystal] or List[str]
        A list of `Crystal` objects. It can also be a list of str as the names
        of the built-in crystals.
    options : ElasticConstraintOptions
        The options of the loss contributed by the constraints.
    weight : float
        The weight of the loss contributed by elastic constants.
    verbose : bool
        If True, key tensors will be logged.

    """
    if options is None:
        options = ElasticConstraintOptions()

    configs = base_nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['energy', 'forces', 'stress']

    with tf.name_scope("Elastic"):

        losses = []
        dtype = get_float_dtype()
        eps = tf.convert_to_tensor(dtype.eps, dtype=dtype, name='eps')
        tau = tf.convert_to_tensor(options.tau, dtype=dtype, name='tau')
        weight = tf.convert_to_tensor(weight, dtype, name='weight/loss')
        f_weight = tf.convert_to_tensor(
            options.forces_weight, dtype, name='weight/f')
        p_weight = tf.convert_to_tensor(
            options.stress_weight, dtype, name='weight/p')
        constraints = {'forces': [eps], 'stress': [eps]}

        for crystal_or_name_or_file in list_of_crystal:
            predictions = []
            labels = []
            cijkl_weights = []
            crystal = get_crystal(crystal_or_name_or_file)

            symbols = set(crystal.atoms.get_chemical_symbols())
            for symbol in symbols:
                if symbol not in base_nn.elements:
                    raise ValueError(f"{symbol} is not supported!")

            with tf.name_scope(f"{crystal.name}/{crystal.phase}"):
                elastic_nn = base_nn.__class__(**configs)
                elastic_clf = base_nn.transformer.as_descriptor_transformer()
                elastic_nn.attach_transformer(elastic_clf)

                features = elastic_clf.get_constant_features(crystal.atoms)

                output = elastic_nn.build(
                    features=features,
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=verbose)
                total_stress = output["total_stress"]
                cell = features["cell"]
                volume = features["volume"]

                with tf.name_scope("Constraints"):
                    constraints['forces'].append(
                        tf.linalg.norm(output["forces"], name='forces'))

                    if options.use_kbar:
                        unit = tf.constant(10.0 / GPa, dtype=total_stress.dtype,
                                           name='unit')
                        value = tf.identity(
                            tf.linalg.norm(
                                tf.math.multiply(output["stress"], unit)),
                            name='kbar')
                    else:
                        unit = tf.constant(1e4 / GPa, dtype=total_stress.dtype,
                                           name='unit')
                        value = tf.identity(
                            tf.linalg.norm(
                                tf.math.multiply(output["stress"], unit)),
                            name='bar')
                    constraints['stress'].append(value)

                    if is_first_replica():
                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, value)
                        tf.add_to_collection(GraphKeys.EVAL_METRICS, value)

                with tf.name_scope("Cijkl"):
                    groups = {vi: {} for vi in range(1, 7)}
                    for elastic_constant in crystal.elastic_constants:
                        i, j, k, l = elastic_constant.ijkl
                        vi = voigt_notation(i, j)
                        vj = voigt_notation(k, l)
                        groups[vi][vj] = elastic_constant

                    for vi, igroup in groups.items():
                        i, j = voigt_to_ij(vi, is_py_index=False)
                        pairs = []
                        for elastic_constant in igroup.values():
                            k, l = elastic_constant.ijkl[2:]
                            pairs.append((k, l))
                        ops = _get_cijkl_op(
                            total_stress, cell, volume, i, j, pairs)
                        for (k, l), cijkl in ops.items():
                            vj = voigt_notation(k, l)
                            elastic_constant = groups[vi][vj]
                            cijkl = tf.identity(cijkl, name=f"C{vi}{vj}")
                            if verbose:
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

                            if is_first_replica():
                                tf.add_to_collection(
                                    GraphKeys.TRAIN_METRICS, cijkl)
                                tf.add_to_collection(
                                    GraphKeys.EVAL_METRICS, cijkl)

                # Loss contribution from elastic constants
                with tf.name_scope("Loss"):
                    predictions = tf.stack(predictions, name='predictions')
                    labels = tf.stack(labels, name='labels')
                    cijkl_weights = tf.stack(cijkl_weights, name='weights')
                    diff = tf.math.subtract(predictions, labels)
                    sd = tf.multiply(
                        cijkl_weights,
                        tf.square(diff),
                        name='sd/weighted')
                    mse = tf.reduce_mean(sd, name='mse')
                    mae = tf.reduce_mean(tf.abs(diff), name='mae')
                    gate = tf.nn.relu(mae - tau, name='gate')
                    mse = tf.multiply(mse, gate, name='mse/gate')
                    mse = tf.add(mse, eps, name='mse/safe')
                    rmse = tf.sqrt(mse, name='rmse')
                    losses.append(rmse)

                    if is_first_replica():
                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)

        with tf.name_scope("Loss"):

            # Loss contribution from constraints because the 2-norm of the total
            # forces and stress of the crystal structure should be zero.
            with tf.name_scope("Constraint"):
                f_loss = tf.add_n(constraints['forces'], name='f_loss')
                f_loss = tf.multiply(f_loss, f_weight, name='weighted/f_loss')
                p_loss = tf.add_n(constraints['stress'], name='p_loss')
                p_loss = tf.multiply(p_loss, p_weight, name='weighted/p_loss')
                c_loss = tf.add(p_loss, f_loss, name='loss/weighted')

            with tf.name_scope("Elastic"):
                e_loss = tf.add_n(losses, name='loss')
                e_loss = tf.multiply(weight, e_loss, name='loss/weighted')

            total_loss = tf.add(e_loss, c_loss, name='total_loss')

            if is_first_replica():
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, e_loss)
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, c_loss)

        return total_loss
