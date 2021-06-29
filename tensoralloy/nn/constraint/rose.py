#!coding=utf-8
"""
The Rose Equation of State constraint for training a `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import GPa, kB, eV
from typing import List
from collections import Counter

from tensoralloy.nn.utils import is_first_replica
from tensoralloy.nn.constraint.data import get_crystal
from tensoralloy.nn.dataclasses import RoseLossOptions
from tensoralloy.neighbor import find_neighbor_size_of_atoms, NeighborSize
from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.utils import GraphKeys, ModeKeys
from tensoralloy import atoms_utils
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_batch_transformer(original_clf: BatchUniversalTransformer,
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
    clf : BatchUniversalTransformer
        The newly created transformer for this trajectory.

    """
    configs = original_clf.as_dict()
    cls = configs.pop('class')

    if cls == "BatchUniversalTransformer":
        nij_max = max(map(lambda x: x.nij, sizes))
        nnl_max = max(map(lambda x: x.nnl, sizes))
        if configs["angular"]:
            ij2k_max = max(map(lambda x: x.ij2k, sizes))
            nijk_max = max(map(lambda x: x.nijk, sizes))
        else:
            ij2k_max = 0
            nijk_max = 0
        configs['nij_max'] = nij_max
        configs['nnl_max'] = nnl_max
        configs['ij2k_max'] = ij2k_max
        configs['nijk_max'] = nijk_max
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

    if options is None:
        options = RoseLossOptions()

    with tf.name_scope("Rose"):
        dtype = get_float_dtype()
        zero = tf.constant(0.0, dtype=dtype, name='zero')

        losses = []

        from tensoralloy.nn.basic import BasicNN
        assert isinstance(base_nn, BasicNN)

        prop = base_nn.variational_energy
        configs['export_properties'] = [prop, 'forces', 'stress']
        configs['minimize_properties'] = [prop, 'forces', 'stress']

        for idx, crystal_or_name_or_file in enumerate(options.crystals):
            crystal = get_crystal(crystal_or_name_or_file)
            if crystal.bulk_modulus == 0:
                continue

            one = tf.convert_to_tensor(1.0, dtype=dtype, name='one')
            two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')
            three = tf.convert_to_tensor(3.0, dtype=dtype, name='three')
            nine = tf.convert_to_tensor(9.0, dtype=dtype, name='nine')

            scope_name = f"{crystal.name}/{crystal.phase}"
            if crystal.temperature > 0:
                kelvin = np.round(crystal.temperature * eV / kB, 0)
                scope_name = f"{scope_name}/{kelvin:.0f}K"

            with tf.name_scope(scope_name):

                nn = base_nn.__class__(**configs)

                with tf.name_scope("Eq"):
                    base_clf = base_nn.transformer
                    rc = base_clf.rc
                    angular = base_clf.angular

                    if isinstance(base_clf, BatchUniversalTransformer):
                        clf = base_clf.as_descriptor_transformer()
                    else:
                        raise ValueError(
                            "A `BatchDescriptorTransformer` must be attached!")
                    nn.attach_transformer(clf)

                    features = clf.get_constant_features(crystal.atoms)
                    output = nn.build(
                        features=features,
                        mode=ModeKeys.PREDICT,
                        verbose=verbose)
                    e0 = tf.identity(output[prop], name='E0')
                    v0 = tf.identity(features["volume"], name='V0')
                    p0 = tf.negative(output['stress'][:3] / GPa,
                                     name='P0')
                    gpa = tf.reduce_mean(p0, name='GPa')
                    pref = tf.convert_to_tensor(options.p_target[idx],
                                                dtype=dtype, name='P')
                    if options.E_target:
                        ecoh = tf.convert_to_tensor(options.E_target[idx],
                                                    dtype=dtype, name='E')
                    else:
                        ecoh = None

                dx = options.dx
                xlo = options.xlo
                xhi = options.xhi
                num = int((xhi - xlo) / dx) + 1
                beta = options.beta[idx]
                eqx = np.linspace(xlo, xhi, num=num, endpoint=True) - 1.0

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
                        atoms_utils.set_electron_temperature(
                            atoms, crystal.temperature)
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
                        fixed_batch = dict()
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
                        mode=ModeKeys.TRAIN,
                        verbose=verbose)

                    predictions = tf.identity(
                        outputs[prop], name='predictions')

                    with tf.name_scope("Ei"):
                        c12 = tf.math.add(one, ax, name='c12')
                        div = tf.math.divide(two * x + three,
                                             tf.pow(x + one, two), 'div')
                        c3 = tf.math.multiply(
                            beta, tf.pow(ax, three) * div, name='c3')
                        coef = tf.math.multiply(tf.exp(-ax), c12 + c3, 'coef')
                        if ecoh is None:
                            labels = tf.multiply(e0, coef, name='labels')
                        else:
                            labels = tf.multiply(ecoh, coef, name='labels')

                    with tf.name_scope("Loss"):
                        ploss = tf.norm(p0 - pref, name='loss/GPa')
                        if ecoh is not None:
                            eloss = tf.abs(e0 - ecoh, name='loss/E')
                        else:
                            eloss = zero

                        diff = tf.math.subtract(predictions, labels, 'diff')
                        mae = tf.reduce_mean(tf.math.abs(diff), name='mae')

                        sds = tf.reduce_sum(tf.math.square(diff), name='sds')
                        eps = tf.convert_to_tensor(
                            get_float_dtype().eps, dtype, 'eps')

                        weight = tf.convert_to_tensor(
                            options.weight, dtype, name='weight')
                        residual = tf.sqrt(sds + eps, name='residual')
                        loss = tf.multiply(residual + ploss + eloss,
                                           weight, name='loss')
                        losses.append(loss)

                    if is_first_replica():
                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)
                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, mae)
                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, gpa)
                        tf.add_to_collection(GraphKeys.TRAIN_METRICS, e0)
                        tf.add_to_collection(GraphKeys.EVAL_METRICS, residual)

        return tf.add_n(losses, name='total_loss')
