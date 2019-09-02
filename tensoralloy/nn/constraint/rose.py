#!coding=utf-8
"""
The Rose Equation of State constraint for training a `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import GPa
from typing import List
from collections import Counter
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.nn.constraint.data import get_crystal
from tensoralloy.nn.dataclasses import RoseLossOptions
from tensoralloy.neighbor import find_neighbor_size_of_atoms, NeighborSize
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.utils import AttributeDict, GraphKeys
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_batch_transformer(original_clf: BatchDescriptorTransformer,
                          trajectory: List[Atoms],
                          sizes: List[NeighborSize],
                          max_occurs: Counter):
    """
    Return a `BatchDescriptorTransformer` for the trajectory.

    Parameters
    ----------
    original_clf : BatchDescriptorTransformer
        The original batch descriptor transformer.
    trajectory : List[Atoms]
        A list of scaled `Atoms` object.
    sizes : List[NeighborSize]
        The corresponding `NeighborSize` for each `Atoms`.
    max_occurs : Counter
        The max occurances of the elements.

    Returns
    -------
    clf : BatchDescriptorTransformer
        The newly created batch descriptor transformer for this trajectory.

    """
    configs = original_clf.as_dict()
    cls = configs.pop('class')

    if cls == 'BatchSymmetryFunctionTransformer':
        nij_max = max(map(lambda x: x.nij, sizes))
        if configs['angular']:
            nijk_max = max(map(lambda x: x.nijk, sizes))
        else:
            nijk_max = 0
        configs['nij_max'] = nij_max
        configs['nijk_max'] = nijk_max

    elif cls == "BatchEAMTransformer" or cls == "BatchADPTransformer":
        nij_max = max(map(lambda x: x.nij, sizes))
        nnl_max = max(map(lambda x: x.nnl, sizes))
        configs['nij_max'] = nij_max
        configs['nnl_max'] = nnl_max

    else:
        raise ValueError(f"Unsupported batch transformer: {cls}")

    # Make sure every element appears in this dict.
    for element in original_clf.elements:
        max_occurs[element] = max(max_occurs[element], 1)

    configs['max_occurs'] = max_occurs
    configs['use_forces'] = False
    configs['use_stress'] = False
    configs['batch_size'] = len(trajectory)

    return original_clf.__class__(**configs)


def get_rose_constraint_loss(base_nn,
                             options: RoseLossOptions = None,
                             verbose=True) -> tf.Tensor:
    """
    Create a Rose Equation of State constraint. This constraint is used to fit
    the bulk modulus.

    Parameters
    ----------
    base_nn : BasicNN
        A `BasicNN`. Its variables will be reused.
    options : RoseLossOptions
        The options for this loss tensor.
    verbose : bool
        If True, key tensors will be logged.

    Returns
    -------
    loss : tf.Tensor
        The total loss of the Rose Equation of State constraint.

    References
    ----------
    Eq.12 of Acta Materialia 52 (2004) 1451–1467.

    """

    configs = base_nn.as_dict()
    configs.pop('class')
    configs['export_properties'] = ['energy']
    configs['minimize_properties'] = ['energy']

    if options is None:
        options = RoseLossOptions()

    with tf.name_scope("Rose/"):

        losses = []

        for idx, crystal_or_name_or_file in enumerate(options.crystals):
            crystal = get_crystal(crystal_or_name_or_file)
            if crystal.bulk_modulus == 0:
                continue

            dtype = get_float_dtype()
            one = tf.convert_to_tensor(1.0, dtype=dtype, name='one')
            two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')
            three = tf.convert_to_tensor(3.0, dtype=dtype, name='three')
            nine = tf.convert_to_tensor(9.0, dtype=dtype, name='nine')

            with tf.name_scope(f"{crystal.name}/{crystal.phase}"):

                nn = base_nn.__class__(**configs)

                with tf.name_scope("Eq"):
                    base_clf = base_nn.transformer
                    rc = base_clf.rc
                    angular = base_clf.angular

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
                    e0 = tf.identity(output["energy"], name='E0')
                    v0 = tf.identity(features["volume"], name='V0')

                dx = options.dx
                delta = options.delta
                beta = options.beta[idx]
                eqx = np.arange(-dx, dx + delta, delta)

                with tf.name_scope("Params"):
                    bulk_modulus = tf.convert_to_tensor(
                        crystal.bulk_modulus * GPa, dtype=dtype, name='B')
                    alpha = tf.sqrt(tf.abs(nine * v0 * bulk_modulus / e0),
                                    name='alpha')
                    x = tf.convert_to_tensor(eqx, dtype=dtype, name='x')
                    ax = tf.math.multiply(alpha, x, name='ax')
                    beta = tf.convert_to_tensor(beta, dtype=dtype, name='beta')

                with tf.name_scope("EOS"):

                    trajectory = []
                    sizes = []
                    max_occurs = Counter()

                    for scale in eqx:
                        atoms = crystal.atoms.copy()
                        atoms.set_cell(crystal.atoms.cell * (1.0 + scale),
                                       scale_atoms=True)
                        atoms.calc = SinglePointCalculator(
                            atoms, **{'energy': 0.0})
                        symbols = atoms.get_chemical_symbols()
                        for el, n in Counter(symbols).items():
                            max_occurs[el] = max(max_occurs[el], n)
                        trajectory.append(atoms)
                        sizes.append(
                            find_neighbor_size_of_atoms(atoms, rc, angular))

                    batch_clf = get_batch_transformer(
                        base_clf, trajectory, sizes, max_occurs)
                    nn.attach_transformer(batch_clf)

                    # Initialize the fixed batch input pipeline
                    with tf.name_scope("Pipeline"):
                        fixed_batch = AttributeDict()
                        decoded_list = []
                        for i, atoms in enumerate(trajectory):
                            with tf.name_scope(f"{i}"):
                                protobuf = tf.convert_to_tensor(
                                    batch_clf.encode(atoms).SerializeToString())
                                decoded_list.append(
                                    batch_clf.decode_protobuf(protobuf))
                        keys = decoded_list[0].keys()
                        for key in keys:
                            fixed_batch[key] = tf.stack(
                                [decoded[key] for decoded in decoded_list],
                                name=f'{key}/batch')

                    outputs = nn.build(
                        features=fixed_batch,
                        mode=tf_estimator.ModeKeys.TRAIN,
                        verbose=verbose)

                    predictions = tf.identity(
                        outputs["energy"], name='predictions')

                    with tf.name_scope("Ei"):
                        c12 = tf.math.add(one, ax, name='c12')
                        div = tf.math.divide(two * x + three,
                                             tf.pow(x + one, two), 'div')
                        c3 = tf.math.multiply(
                            beta, tf.pow(ax, three) * div, name='c3')
                        coef = tf.math.multiply(tf.exp(-ax), c12 + c3, 'coef')
                        labels = tf.multiply(e0, coef, name='labels')

                    with tf.name_scope("Loss"):
                        diff = tf.math.subtract(predictions, labels, 'diff')
                        mae = tf.reduce_mean(tf.math.abs(diff), name='mae')

                        sds = tf.reduce_sum(tf.math.square(diff), name='sds')
                        eps = tf.convert_to_tensor(
                            get_float_dtype().eps, dtype, 'eps')
                        weight = tf.convert_to_tensor(
                            options.weight, dtype, name='weight')
                        residual = tf.sqrt(sds + eps, name='residual')
                        loss = tf.multiply(residual, weight, name='loss')
                        losses.append(loss)

                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)
                    tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)
                    tf.add_to_collection(GraphKeys.EVAL_METRICS, residual)

        return tf.add_n(losses, name='total_loss')
