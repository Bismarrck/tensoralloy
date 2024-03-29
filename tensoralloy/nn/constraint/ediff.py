#!coding=utf-8
"""
The energy difference constraint.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.nn.constraint.data import get_crystal, Crystal
from tensoralloy.nn.dataclasses import EnergyDifferenceLossOptions
from tensoralloy.precision import get_float_dtype
from tensoralloy.utils import GraphKeys, ModeKeys
from tensoralloy.nn.utils import is_first_replica, logcosh

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def may_merge_constraints(options: EnergyDifferenceLossOptions):
    """
    Merge constraints.
    """
    n = len(options.diff)
    tasks = {}
    for i in range(n):
        key = options.references[i]
        tasks[key] = tasks.get(key, []) + [(options.crystals[i],
                                            options.diff[i])]
    return tasks


def calculate(base_nn, crystal: Crystal, verbose=False):
    """
    Return the total energy Op of the given crystal.
    """
    configs = base_nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['energy', 'forces']
    configs['minimize_properties'] = ['energy']

    nn = base_nn.__class__(**configs)
    base_clf = base_nn.transformer
    assert isinstance(base_clf, BatchDescriptorTransformer)
    clf = base_clf.as_descriptor_transformer()
    nn.attach_transformer(clf)
    features = clf.get_constant_features(crystal.atoms)
    output = nn.build(
        features=features,
        mode=ModeKeys.PREDICT,
        verbose=verbose)
    energy = output["energy"]
    fnorm = tf.linalg.norm(output['forces'], name='fnorm')
    natoms = tf.constant(len(crystal.atoms), dtype=energy.dtype, name='natoms')
    return tf.math.truediv(energy, natoms, name='eatom'), fnorm


def get_energy_difference_constraint_loss(
        base_nn,
        options: EnergyDifferenceLossOptions = None,
        verbose=False):
    """
    Return the total loss of all energy difference constraints.
    """
    if options is None:
        options = EnergyDifferenceLossOptions()

    dtype = get_float_dtype()

    with tf.name_scope("Diff"):
        y_true = []
        y_pred = []
        losses = []
        for reference, pairs in may_merge_constraints(options).items():
            reference = get_crystal(reference)
            with tf.name_scope(f"{reference.name}/{reference.phase}"):
                fnorms = []
                e0 = calculate(base_nn=base_nn,
                               crystal=reference,
                               verbose=verbose)[0]
                for (crystal, ediff) in pairs:
                    crystal = get_crystal(crystal)
                    with tf.name_scope(f"{crystal.name}/{crystal.phase}"):
                        ei, fnorm = calculate(base_nn=base_nn,
                                           crystal=crystal,
                                           verbose=verbose)
                        val = tf.math.subtract(ei, e0, name='pred')
                        y_true.append(
                            tf.constant(ediff, dtype=dtype, name="diff"))
                        y_pred.append(val)
                        fnorms.append(fnorm)
                        if is_first_replica():
                            tf.add_to_collection(GraphKeys.TRAIN_METRICS, fnorm)
                            tf.add_to_collection(GraphKeys.EVAL_METRICS, val)

                with tf.name_scope("Loss"):
                    y_true = tf.stack(y_true, name='y_true')
                    y_pred = tf.stack(y_pred, name='y_pred')
                    y_diff = tf.math.subtract(y_pred, y_true, name='y_diff')
                    mae = tf.reduce_mean(tf.math.abs(y_diff), name='mae')
                    if options.method == "mae":
                        loss = mae
                    else:
                        loss = tf.reduce_mean(
                            logcosh(y_diff, dtype), name='logcosh')
                    if is_first_replica():
                        tf.add_to_collection(GraphKeys.EVAL_METRICS, mae)
                    weight = tf.convert_to_tensor(
                        options.weight, dtype, name='weight')
                    loss = tf.math.multiply(loss, weight,
                                            name=f'{options.method}/weighted')
                    fnorm_weight = tf.convert_to_tensor(
                        options.forces_weight, dtype=dtype, name='weight/fnorm')
                    residual = tf.add_n(fnorms, name='residual')
                    residual = tf.math.multiply(
                        residual, fnorm_weight, name='residual/weighted')
                    loss = tf.math.add(loss, residual, name='loss')
                    if is_first_replica():
                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)
                    losses.append(loss)
        return tf.add_n(losses, name='loss')
