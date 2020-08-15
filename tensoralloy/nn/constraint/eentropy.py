#!coding=utf-8
"""
The electron entropy constraint.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import warnings

from tensorflow_estimator import estimator as tf_estimator
from ase.units import kB

from tensoralloy.nn.constraint.data import get_crystal
from tensoralloy.nn.dataclasses import EEntropyConstraintOptions
from tensoralloy.precision import get_float_dtype
from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.nn.utils import is_first_replica
from tensoralloy.utils import GraphKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_eentropy_constraint_loss(base_nn,
                                 options: EEntropyConstraintOptions,
                                 verbose=False):
    """
    Return the simple electron entropy constraint.
    """
    from tensoralloy.nn.atomic import AtomicNN
    if not isinstance(base_nn, AtomicNN):
        warnings.warn(
            "Pair style 'atomic' is required for the eentropy constraint")
        return None
    if not base_nn.finite_temperature_options.algorithm == 'full':
        warnings.warn("The finite temperature algorithm 'full' is required "
                      "for the eentropy constraint")
        return None

    if options is None:
        options = EEntropyConstraintOptions()

    configs = base_nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['eentropy']
    configs['minimize_properties'] = ['eentropy']

    dtype = get_float_dtype()

    with tf.name_scope("EENTRO"):
        y_true = []
        y_pred = []

        for crystal_or_name_or_file in options.crystals:
            crystal = get_crystal(crystal_or_name_or_file)
            kelvin = crystal.temperature * kB
            scope = f"{crystal.name}/{crystal.phase}/{kelvin:.0f}K"

            with tf.name_scope(scope):
                nn = base_nn.__class__(**configs)
                base_clf = base_nn.transformer
                if isinstance(base_clf, BatchUniversalTransformer):
                    clf = base_clf.as_descriptor_transformer()
                else:
                    clf = base_clf
                nn.attach_transformer(clf)
                features = clf.get_constant_features(crystal.atoms)
                output = nn.build(
                    features=features,
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=verbose)
                eentropy = output['eentropy']
                y_true.append(
                    tf.constant(crystal.eentropy, dtype=dtype, name='eentropy'))
                y_pred.append(eentropy)

        with tf.name_scope("Loss"):
            y_true = tf.stack(y_true, name='y_true')
            y_pred = tf.stack(y_pred, name='y_pred')
            y_diff = tf.math.subtract(y_pred, y_true, name='y_diff')
            loss = tf.reduce_mean(tf.math.abs(y_diff), name='mae')
            if is_first_replica():
                tf.add_to_collection(GraphKeys.EVAL_METRICS, loss)
            weight = tf.convert_to_tensor(
                options.weight, dtype, name='weight')
            loss = tf.math.multiply(loss, weight,
                                    name=f'loss/weighted')
            if is_first_replica():
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)
        return loss
