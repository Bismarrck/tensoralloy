#!coding=utf-8
"""
The vacancy formation energy constraint.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.nn.dataclasses import VacancyLossOptions
from tensoralloy.nn.constraint.data import get_crystal
from tensoralloy.utils import GraphKeys
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_vacancy_formation_energy_loss(base_nn,
                                      options: VacancyLossOptions = None,
                                      verbose=True) -> tf.Tensor:
    """
    Return the total loss tensor of the vacancy formation energy constraint.
    """
    configs = base_nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['energy', 'forces']
    configs['minimize_properties'] = ['energy', 'forces']

    if options is None:
        options = VacancyLossOptions()

    with tf.name_scope("Vacancy/"):

        losses = []

        dtype = get_float_dtype()
        weight = tf.convert_to_tensor(
            options.weight, name='weight', dtype=dtype)

        for idx, crystal_or_name_or_file in enumerate(options.crystals):
            crystal = get_crystal(crystal_or_name_or_file)
            if crystal.vacancy_atoms is None:
                continue

            with tf.name_scope(f"{crystal.name}/{crystal.phase}"):

                nn = base_nn.__class__(**configs)

                with tf.name_scope("Eq"):
                    base_clf = base_nn.transformer

                    if isinstance(base_clf, BatchDescriptorTransformer):
                        clf = base_clf.as_descriptor_transformer()
                    else:
                        raise ValueError(
                            "A `BatchDescriptorTransformer` must be attached!")
                    nn.attach_transformer(clf)

                    features = clf.get_constant_features(crystal.atoms)
                    output = nn.build(
                        features=features,
                        mode=tf_estimator.ModeKeys.PREDICT,
                        verbose=verbose)

                    n0 = tf.convert_to_tensor(
                        len(crystal.atoms), dtype=dtype, name='natoms')
                    e0 = tf.identity(output["energy"], name='E0')

                with tf.name_scope("Vacancy"):
                    features = clf.get_constant_features(crystal.vacancy_atoms)
                    output = nn.build(
                        features=features,
                        mode=tf_estimator.ModeKeys.PREDICT,
                        verbose=verbose)
                    n1 = tf.convert_to_tensor(
                        len(crystal.vacancy_atoms), dtype=dtype, name='natoms')
                    e1 = tf.identity(output["energy"], name='E1')
                    fnorm1 = tf.linalg.norm(output["forces"], name="Fnorm")

                vfe_true = tf.convert_to_tensor(
                    crystal.vacancy_formation_energy,
                    dtype=dtype,
                    name='vfe/true')

                vfe = tf.math.subtract(
                    e1, tf.math.divide(n1, n0) * e0, name='vfe')

                mae = tf.math.abs(vfe - vfe_true, name='mae')
                mae = tf.math.multiply(mae, weight, name='mae/weighted')

                if options.forces_weight > 0.0:
                    loss = tf.math.add(mae, fnorm1, name='loss')
                else:
                    loss = tf.identity(mae, name='loss')

                tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, vfe)

                losses.append(loss)

        return tf.add_n(losses, name='total_loss')
