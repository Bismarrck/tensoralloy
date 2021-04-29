#!coding=utf-8
"""
The force constants constraint.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from typing import List, Union

from tensoralloy.nn.dataclasses import ForceConstantsLossOptions
from tensoralloy.nn.constraint.data import Crystal, get_crystal
from tensoralloy.nn.utils import is_first_replica
from tensoralloy.transformer.vap import VirtualAtomMap
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.utils import ModeKeys, GraphKeys
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def reorder_phonopy_fc2(fc2, vap):
    """
    Convert the Phonopy force constants array to VAP form.

    Parameters
    ----------
    fc2 : array_like
        The [N, N, 3, 3] force constants array used by Phonopy.
    vap : VirtualAtomMap
        The corresponding VirtualAtomMap.

    Returns
    -------
    fc2 : array_like
        The [N_vap, 3, N_vap, 3] force constants array and associated weights.
    weights : array_like
        The [N_vap, 3, N_vap, 3] array indicating the weights of force
        constants.

    """
    n_vap = vap.max_vap_natoms
    n = fc2.shape[0]
    dtype = get_float_dtype().as_numpy_dtype
    fc2_vap = np.zeros((n_vap, 3, n_vap, 3), dtype=dtype)
    weights = np.zeros_like(fc2_vap, dtype=dtype)
    for i in range(n):
        ii = vap.local_to_gsl_map[i + 1]
        for alpha in range(3):
            for j in range(n):
                jj = vap.local_to_gsl_map[j + 1]
                for beta in range(3):
                    fc2_vap[ii, alpha, jj, beta] = fc2[i, j, alpha, beta]
                    weights[ii, alpha, jj, beta] = 1.0
    return fc2_vap, weights


def get_fc2_loss(base_nn,
                 list_of_crystal: List[Union[Crystal, str]],
                 options: ForceConstantsLossOptions = None,
                 weight=1.0,
                 verbose=True):
    """
    Return a special loss: RMSE (GPa) of force constants of crystals.

    Parameters
    ----------
    base_nn : BasicNN
        A `BasicNN`. Its weights will be reused.
    list_of_crystal : List[Crystal] or List[str]
        A list of `Crystal` objects. It can also be a list of str as the names
        of the built-in crystals.
    options : ForceConstantsLossOptions
        The options of the loss contributed by the constraints.
    weight : float
        The weight of the loss contributed by force constants.
    verbose : bool
        If True, key tensors will be logged.

    """
    if options is None:
        options = ForceConstantsLossOptions()

    configs = base_nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['energy', 'forces', 'hessian']

    with tf.name_scope("FC"):

        losses = []
        dtype = get_float_dtype()
        eps = tf.convert_to_tensor(dtype.eps, dtype=dtype, name='eps')
        weight = tf.convert_to_tensor(weight, dtype, name='weight/loss')
        f_weight = tf.convert_to_tensor(
            options.forces_weight, dtype, name='weight/f')
        constraints = {'forces': [eps]}

        for crystal_or_name_or_file in list_of_crystal:
            crystal = get_crystal(crystal_or_name_or_file)

            symbols = set(crystal.atoms.get_chemical_symbols())
            for symbol in symbols:
                if symbol not in base_nn.elements:
                    raise ValueError(f"{symbol} is not supported!")

            with tf.name_scope(f"{crystal.name}/{crystal.phase}"):
                fc2_nn = base_nn.__class__(**configs)

                if isinstance(base_nn.transformer, UniversalTransformer):
                    fc2_clf = base_nn.transformer
                else:
                    fc2_clf = base_nn.transformer.as_descriptor_transformer()
                fc2_nn.attach_transformer(fc2_clf)
                features = fc2_clf.get_constant_features(crystal.supercell)
                output = fc2_nn.build(
                    features=features,
                    mode=ModeKeys.PREDICT,
                    verbose=verbose)
                fc2 = output["hessian"]
                fc2_ref, weights = reorder_phonopy_fc2(
                    crystal.fc2, fc2_clf.get_vap_transformer(crystal.supercell))
                fc2_ref = tf.convert_to_tensor(
                    fc2_ref, dtype=dtype, name='fc2/ref')
                weights_ref = tf.convert_to_tensor(
                    weights, dtype=dtype, name="fc2/weights")
                with tf.name_scope("Constraints"):
                    constraints['forces'].append(
                        tf.linalg.norm(output["forces"], name='forces'))

                diff = tf.multiply(weights_ref, fc2 - fc2_ref, name='diff')
                mae = tf.reduce_mean(tf.abs(diff), name='mae')
                rmse = tf.sqrt(tf.reduce_mean(tf.square(diff)), name='rmse')
                losses.append(rmse)

                if is_first_replica():
                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)

        with tf.name_scope("Loss"):
            # Loss contribution from constraints because the 2-norm of the total
            # forces and stress of the crystal structure should be zero.
            with tf.name_scope("Constraint"):
                f_loss = tf.add_n(constraints['forces'], name='f_loss')
                f_loss = tf.multiply(f_loss, f_weight, name='weighted/f_loss')
            with tf.name_scope("FC"):
                fc_loss = tf.add_n(losses, name='loss')
                fc_loss = tf.multiply(weight, fc_loss, name='loss/weighted')
            total_loss = tf.add(f_loss, fc_loss, name='total_loss')
            if is_first_replica():
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, fc_loss)

        return total_loss
