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
from ase.db import connect
from os.path import join
from collections import Counter
from typing import List
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

from tensoralloy.io.neighbor import find_neighbor_size_of_atoms
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.atomic import AtomicNN
from tensoralloy.nn.dataclasses import LossParameters
from tensoralloy.transformer import BatchSymmetryFunctionTransformer
from tensoralloy.utils import AttributeDict, Defaults
from tensoralloy.test_utils import assert_array_equal, datasets_dir, test_dir
from tensoralloy.dtypes import set_float_precision

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
    nij_max, _, _ = find_neighbor_size_of_atoms(atoms, rc, k_max=2)
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
            nn = AtomicNN(elements, minimize_properties=list_of_properties)
            clf = _get_transformer()
            nn.attach_transformer(clf)
            protobuf = tf.convert_to_tensor(
                clf.encode(atoms).SerializeToString())
            example = clf.decode_protobuf(protobuf)
            batch = AttributeDict()
            for key, tensor in example.items():
                batch[key] = tf.expand_dims(
                    tensor, axis=0, name=tensor.op.name + '/batch')
            labels = AttributeDict(energy=batch.pop('y_true'),
                                   energy_confidence=batch.pop('y_conf'))
            labels['forces'] = batch.pop('f_true')
            labels['forces_confidence'] = batch.pop('f_conf')
            labels['stress'] = batch.pop('stress')
            labels['stress_confidence'] = batch.pop('s_conf')
            labels['total_pressure'] = batch.pop('total_pressure')

            loss_parameters = LossParameters()
            loss_parameters.elastic.crystals = ['Ni']
            loss_parameters.elastic.weight = 0.1
            loss_parameters.elastic.constraint.forces_weight = 1.0
            loss_parameters.elastic.constraint.stress_weight = 0.1
            loss_parameters.elastic.constraint.use_kbar = True

            mode = tf_estimator.ModeKeys.TRAIN

            try:
                predictions = nn.build(batch, mode=mode, verbose=True)
            except Exception:
                return False
            try:
                nn.get_total_loss(predictions=predictions,
                                  labels=labels,
                                  n_atoms=batch.n_atoms,
                                  loss_parameters=loss_parameters,
                                  mode=mode)
            except Exception as excp:
                print(excp)
                return False
            else:
                return True

    set_float_precision('medium')

    for case in (['energy', 'elastic'],
                 ['energy', 'stress'],
                 ['energy', 'elastic']):
        assert_true(_test_with_properties(case), msg=f"{case} is failed")

    set_float_precision('high')


if __name__ == "__main__":
    nose.run()
