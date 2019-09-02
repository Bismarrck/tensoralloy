# coding=utf-8
"""
This module defines unit tests of `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from tensorflow_estimator import estimator as tf_estimator
from nose.tools import assert_dict_equal, assert_equal, assert_true
from nose.tools import assert_almost_equal
from ase.db import connect
from ase.build import bulk
from ase.units import GPa
from os.path import join
from collections import Counter
from typing import List
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.atomic import AtomicNN
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.nn.dataclasses import LossParameters
from tensoralloy.transformer import BatchSymmetryFunctionTransformer
from tensoralloy.transformer import EAMTransformer
from tensoralloy.utils import AttributeDict, Defaults, set_pulay_stress
from tensoralloy.test_utils import assert_array_equal, datasets_dir, test_dir
from tensoralloy.test_utils import assert_array_almost_equal
from tensoralloy.precision import precision_scope

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_energy_pv():
    """
    Test the total energy Op and the stress Op after the introduction of `E_pv`.
    """
    atoms = bulk('Ni')
    volume = atoms.get_volume()
    rc = 6.0
    elements = ['Ni']
    mode = tf_estimator.ModeKeys.PREDICT

    with precision_scope("high"):

        with tf.Graph().as_default():

            nn = EamAlloyNN(elements, custom_potentials='zjw04',
                            export_properties=['energy', 'forces', 'stress'])
            clf = EAMTransformer(rc, elements)
            nn.attach_transformer(clf)

            with tf.name_scope("0GPa"):
                zero_ops = nn.build(
                    clf.get_constant_features(atoms), mode=mode, verbose=False)

            with tf.name_scope("10GPa"):
                set_pulay_stress(atoms, 10.0 * GPa)
                ten_ops = nn.build(
                    clf.get_constant_features(atoms), mode=mode, verbose=False)

            with tf.name_scope("100GPa"):
                set_pulay_stress(atoms, 100.0 * GPa)
                hundred_ops = nn.build(
                    clf.get_constant_features(atoms), mode=mode, verbose=False)

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                zero, ten, hundred = sess.run([zero_ops, ten_ops, hundred_ops])

            assert_almost_equal(zero["energy"], hundred["energy"], delta=1e-8)
            assert_almost_equal(zero["enthalpy"] - hundred["enthalpy"],
                                -100 * GPa * volume, delta=1e-8)

            # Only diag elements are affected by pulay stress.
            assert_array_almost_equal(
                ten["stress"][:3] / GPa - hundred["stress"][:3] / GPa,
                np.ones(3) * 90.0,
                delta=1e-8)

            assert_array_almost_equal(
                ten["stress"][3:] / GPa - hundred["stress"][3:] / GPa,
                np.zeros(3),
                delta=1e-8)


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


def test_graph_model_variables():
    """
    Test the exported variables of `BasicNN.export`.
    """
    save_dir = join(test_dir(), "checkpoints", "qm7-k2")

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(
            join(save_dir, "model.ckpt-10000.meta"))

        with tf.Session() as sess:
            saver.restore(sess, join(save_dir, "model.ckpt-10000"))
            train_values = {}
            for var in tf.trainable_variables():
                train_values[var.op.name] = sess.run(
                    graph.get_tensor_by_name(
                        f"{var.op.name}/ExponentialMovingAverage:0"))


    graph_path = join(save_dir, "qm7.pb")
    graph = tf.Graph()

    with graph.as_default():

        output_graph_def = graph_pb2.GraphDef()
        with open(graph_path, "rb") as fp:
            output_graph_def.ParseFromString(fp.read())
            importer.import_graph_def(output_graph_def, name="")

        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True),
            graph=graph)

        with sess:
            for var_op_name, ref_value in train_values.items():
                value = sess.run(f"{var_op_name}:0")
                dmax = np.abs(np.atleast_1d(value - ref_value)).max()
                assert_equal(dmax, 0)


def test_build_nn_with_properties():
    """
    Test the method `BasicNN.build` with different `minimize_properties`.
    """
    elements = ['Mo', 'Ni']
    rc = 6.5
    batch_size = 1
    db = connect(join(datasets_dir(), 'snap.db'))
    atoms = db.get_atoms(id=1)
    nij_max = find_neighbor_size_of_atoms(atoms, rc).nij
    max_occurs = Counter(atoms.get_chemical_symbols())

    def _get_transformer():
        """
        A helper function to return a `BatchSymmetryFunctionTransformer`.
        """
        return BatchSymmetryFunctionTransformer(
            rc=rc, max_occurs=max_occurs, nij_max=nij_max, nijk_max=0,
            batch_size=batch_size, use_stress=True, use_forces=True)

    def _test_with_properties(list_of_properties: List[str]):
        """
        Run a test.
        """
        with tf.Graph().as_default():
            nn = AtomicNN(elements=elements,
                          minmax_scale=False,
                          minimize_properties=list_of_properties)
            clf = _get_transformer()
            nn.attach_transformer(clf)
            protobuf = tf.convert_to_tensor(
                clf.encode(atoms).SerializeToString())
            example = clf.decode_protobuf(protobuf)
            batch = dict()
            for key, tensor in example.items():
                batch[key] = tf.expand_dims(
                    tensor, axis=0, name=tensor.op.name + '/batch')
            labels = dict(energy=batch.pop('y_true'))
            labels['forces'] = batch.pop('f_true')
            labels['stress'] = batch.pop('stress')
            labels['total_pressure'] = batch.pop('total_pressure')

            loss_parameters = LossParameters()
            loss_parameters.elastic.crystals = ['Ni']
            loss_parameters.elastic.weight = 0.1
            loss_parameters.elastic.constraint.forces_weight = 1.0
            loss_parameters.elastic.constraint.stress_weight = 0.1
            loss_parameters.elastic.constraint.use_kbar = True

            mode = tf_estimator.ModeKeys.TRAIN

            tf.train.get_or_create_global_step()

            try:
                predictions = nn.build(batch, mode=mode, verbose=True)
            except Exception as excp:
                print(excp)
                return False
            try:
                nn.get_total_loss(predictions=predictions,
                                  labels=labels,
                                  n_atoms=batch["n_atoms"],
                                  atom_masks=batch["atom_masks"],
                                  loss_parameters=loss_parameters,
                                  mode=mode)
            except Exception as excp:
                print(excp)
                return False
            else:
                return True

    with precision_scope('medium'):
        for case in (['energy', 'elastic'],
                     ['energy', 'stress'],
                     ['energy', 'elastic']):
            assert_true(_test_with_properties(case), msg=f"{case} is failed")


if __name__ == "__main__":
    nose.run()
