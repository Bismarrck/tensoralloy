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
from typing import List, Union
from collections import Counter
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.constraint.data import Crystal, get_crystal
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
            nijk_max = max(map(lambda x: x.nij, sizes))
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


def get_rose_constraint_loss(base_nn: BasicNN,
                             list_of_crystal: List[Union[str, Crystal]],
                             beta: List[float],
                             dx=0.10,
                             delta=0.01,
                             weight=1.0) -> tf.Tensor:
    """
    Create a Rose Equation of State constraint. This constraint is used to fit
    the bulk modulus.

    Parameters
    ----------
    base_nn : BasicNN
        A `BasicNN`. Its variables will be reused.
    list_of_crystal : List[Union[str, Crystal]]
        A list of `Crystal` objects. It can also be a list of str as the names
        of the built-in crystals.
    beta : List[float]
        The adjustable parameter for each crystal.
    dx : float
        The volume scaling range, 0.30 <= dx <= 0.01.
    delta : float
        The delta between two adjacents. n_total = int(2 * dx / delta) + 1.
    weight : float
        The loss weight.

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

    for crystal_or_name_or_file in list_of_crystal:
        crystal = get_crystal(crystal_or_name_or_file)
        if crystal.bulk_modulus == 0:
            continue

        with tf.name_scope(f"Rose/{crystal.name}/"):

            nn = base_nn.__class__(**configs)

            with tf.name_scope("Equilibrium"):
                base_clf = base_nn.transformer
                rc = base_clf.rc
                angular = base_clf.k_max == 3

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
                    verbose=True)
                e0 = tf.identity(output.energy, name='E0')
                v0 = tf.identity(features.volume, name='v0')

            dtype = e0.dtype
            eqx = np.arange(-dx, dx + delta, delta)

            with tf.name_scope("Params"):
                b = tf.convert_to_tensor(crystal.bulk_modulus * GPa,
                                         dtype=dtype, name='B')
                alpha = tf.sqrt(tf.abs(9.0 * v0 * b / e0), name='alpha')
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
                    atoms.calc = SinglePointCalculator(atoms, **{'energy': 0.0})
                    for el, n in Counter(atoms.get_chemical_symbols()).items():
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
                    verbose=False)

                predictions = tf.identity(outputs.energy, name='predictions')

                with tf.name_scope("Ei"):
                    c12 = 1 + ax
                    c3 = beta * tf.pow(ax, 3) * (2 * x + 3) / tf.pow(x + 1, 2)
                    coef = tf.exp(-ax) * (c12 + c3)
                    labels = tf.multiply(e0, coef, name='labels')

                with tf.name_scope("Loss"):
                    diff = tf.math.subtract(predictions, labels, name='diff')
                    mae = tf.reduce_mean(tf.math.abs(diff), name='mae')
                    mse = tf.reduce_mean(tf.math.square(diff), name='mse')
                    eps = tf.convert_to_tensor(
                        get_float_dtype().eps, dtype, 'eps')
                    weight = tf.convert_to_tensor(weight, dtype, 'weight')
                    rmse = tf.sqrt(mse + eps, name='rmse')
                    loss = tf.multiply(rmse, weight, name='loss')

                tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)

                return loss