# coding=utf-8
"""
This module defines unit tests of `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_dict_equal, assert_equal
from ase.db import connect
from os.path import join
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

from tensoralloy.nn.basic import BasicNN
from tensoralloy.utils import AttributeDict, Defaults
from tensoralloy.test_utils import assert_array_equal, datasets_dir, test_dir

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


def test_check_hparams():
    """
    Test the static method `BasicNN._check_hparams`.
    """
    nn = BasicNN(elements=['Ni'], activation='leaky_relu',
                 minimize_properties=['stress'], export_properties=['stress'])
    
    defaults = AttributeDict(
        loss=AttributeDict(
            equivalently_trusted=True,
            energy=AttributeDict(weight=1.0, per_atom_loss=False),
            forces=AttributeDict(weight=1.0),
            stress=AttributeDict(weight=1.0, use_rmse=True),
            total_pressure=AttributeDict(weight=1.0),
            l2=AttributeDict(weight=0.01),
            elastic=AttributeDict(weight=0.1, crystals=[],
                                  constraint_weight=10.0)))

    # noinspection PyTypeChecker
    hparams = nn._check_loss_hparams(None)
    assert_dict_equal(defaults, hparams)
    
    hparams = nn._check_loss_hparams(AttributeDict())
    assert_dict_equal(defaults, hparams)

    hparams = nn._check_loss_hparams(AttributeDict(loss={}))
    assert_dict_equal(defaults, hparams)

    hparams = nn._check_loss_hparams(
        AttributeDict(loss={'energy': {'weight': 3.0},
                            'equivalently_trusted': False}))
    assert_equal(hparams.loss.energy.weight, 3.0)
    assert_equal(hparams.loss.forces.weight, 1.0)
    assert_equal(hparams.loss.equivalently_trusted, False)


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


if __name__ == "__main__":
    nose.run()
