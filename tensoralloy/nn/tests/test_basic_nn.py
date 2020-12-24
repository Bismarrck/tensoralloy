# coding=utf-8
"""
This module defines unit tests of `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_dict_equal, assert_true
from ase.db import connect
from os.path import join
from collections import Counter
from typing import List

from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.atomic import AtomicNN, TemperatureDependentAtomicNN
from tensoralloy.nn.atomic import SymmetryFunction
from tensoralloy.nn.dataclasses import LossParameters
from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.utils import Defaults, ModeKeys
from tensoralloy.test_utils import assert_array_equal, datasets_dir
from tensoralloy.precision import precision_scope


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_hidden_sizes():
    """
    Test setting hidden sizes of `BasicNN`.
    """
    elements = sorted(['Al', 'Cu'])

    nn = BasicNN(elements)
    assert_dict_equal(nn.hidden_sizes, {'Al': Defaults.hidden_sizes,
                                        'Cu': Defaults.hidden_sizes})

    nn = BasicNN(elements, hidden_sizes=32)
    assert_dict_equal(nn.hidden_sizes, {'Al': [32],
                                        'Cu': [32]})

    nn = BasicNN(elements, hidden_sizes=[64, 32])
    assert_dict_equal(nn.hidden_sizes, {'Al': [64, 32],
                                        'Cu': [64, 32]})

    nn = BasicNN(elements, hidden_sizes={'Al': [32, 16]})
    assert_dict_equal(nn.hidden_sizes, {'Al': [32, 16],
                                        'Cu': Defaults.hidden_sizes})

    nn = BasicNN(elements, hidden_sizes={'Al': [32, 32]})
    assert_dict_equal(nn.hidden_sizes, {'Al': [32, 32],
                                        'Cu': Defaults.hidden_sizes})


def test_convert_to_voigt_stress():
    """
    Test the method `BasicNN._convert_to_voigt_stress`.
    """
    db = connect(join(datasets_dir(), 'snap-Ni.db'))
    nn = BasicNN(elements=['Ni'], activation='leaky_relu',
                 minimize_properties=['stress'], export_properties=['stress'])

    with tf.Graph().as_default():

        batch_size = 4
        batch = np.zeros((batch_size, 3, 3), dtype=np.float64)
        batch_voigt = np.zeros((batch_size, 6), dtype=np.float64)

        for i, index in enumerate(range(1, 5)):
            atoms = db.get_atoms(f'id={index}')
            batch[i] = atoms.get_stress(voigt=False)
            batch_voigt[i] = atoms.get_stress(voigt=True)

        op1 = nn._convert_to_voigt_stress(
            tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_size))
        op2 = nn._convert_to_voigt_stress(
            tf.convert_to_tensor(batch[0]), batch_size=None)

        with tf.Session() as sess:
            pred_batch_voigt, pred_voigt = sess.run([op1, op2])

        assert_array_equal(pred_voigt, batch_voigt[0])
        assert_array_equal(pred_batch_voigt, batch_voigt)


def test_build_nn_with_properties():
    """
    Test the method `BasicNN.build` with different `minimize_properties`.
    """
    elements = ['Mo', 'Ni']
    rc = 6.5
    batch_size = 1
    db = connect(join(datasets_dir(), 'snap.db'))
    atoms = db.get_atoms(id=1)
    size = find_neighbor_size_of_atoms(atoms, rc)
    nij_max = size.nij
    nnl_max = size.nnl
    max_occurs = Counter(atoms.get_chemical_symbols())

    def _test_with_properties(list_of_properties: List[str],
                              finite_temperature=False):
        """
        Run a test.
        """
        with tf.Graph().as_default():
            sf = SymmetryFunction(elements=elements)
            clf = BatchUniversalTransformer(
                rcut=rc, max_occurs=max_occurs, nij_max=nij_max,
                nnl_max=nnl_max, batch_size=batch_size, use_stress=True,
                use_forces=True)
            kwargs = dict(
                elements=elements,
                descriptor=sf,
                minmax_scale=False,
                minimize_properties=list_of_properties)
            if finite_temperature:
                nn = TemperatureDependentAtomicNN(**kwargs)
            else:
                nn = AtomicNN(**kwargs)
            nn.attach_transformer(clf)
            protobuf = tf.convert_to_tensor(
                clf.encode(atoms).SerializeToString())
            example = clf.decode_protobuf(protobuf)
            batch = dict()
            for key, tensor in example.items():
                batch[key] = tf.expand_dims(
                    tensor, axis=0, name=tensor.op.name + '/batch')
            labels = dict(energy=batch.pop('energy'))
            labels['forces'] = batch.pop('forces')
            labels['stress'] = batch.pop('stress')
            labels['total_pressure'] = batch.pop('total_pressure')

            loss_parameters = LossParameters()
            loss_parameters.elastic.crystals = ['Ni']
            loss_parameters.elastic.weight = 0.1
            loss_parameters.elastic.constraint.forces_weight = 1.0
            loss_parameters.elastic.constraint.stress_weight = 0.1
            loss_parameters.elastic.constraint.use_kbar = True

            mode = ModeKeys.TRAIN

            tf.train.get_or_create_global_step()

            try:
                predictions = nn.build(batch, mode=mode, verbose=True)
            except Exception as excp:
                print(excp)
                return False
            try:
                nn.get_total_loss(predictions=predictions,
                                  labels=labels,
                                  n_atoms=batch["n_atoms_vap"],
                                  atom_masks=batch["atom_masks"],
                                  loss_parameters=loss_parameters,
                                  mode=mode)
            except Exception as excp:
                print(excp)
                return False
            else:
                return True

    with precision_scope('medium'):
        for case, temperature in ((['energy', 'elastic'], False),
                                  (['energy', 'stress'], False),
                                  (['free_energy', 'eentropy'], True)):
            assert_true(_test_with_properties(case), msg=f"{case} is failed")


if __name__ == "__main__":
    nose.run()
