# coding=utf-8
"""
This module defines unit tests of `EamNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_equal, assert_list_equal, assert_tuple_equal

from ..eam import EamNN
from tensoralloy.misc import AttributeDict
from tensoralloy.test_utils import assert_array_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TestData:
    """
    A private data container for unit tests of this module.
    """

    def __init__(self):
        """
        Initialization method.
        """
        self.batch_size = 10
        self.max_n_terms = 2
        self.max_n_al = 5
        self.max_n_cu = 7
        self.max_n_atoms = self.max_n_al + self.max_n_cu + 1
        self.nnl = 10
        self.elements = sorted(['Cu', 'Al'])

        shape_al = (self.batch_size, self.max_n_terms, self.max_n_al, self.nnl)
        shape_cu = (self.batch_size, self.max_n_terms, self.max_n_cu, self.nnl)

        self.g_al = np.random.randn(*shape_al)
        self.m_al = np.random.randint(0, 2, shape_al).astype(np.float64)
        self.g_cu = np.random.randn(*shape_cu)
        self.m_cu = np.random.randint(0, 2, shape_cu).astype(np.float64)

        self.y_alal = np.random.randn(self.batch_size, self.max_n_al)
        self.y_alcu = np.random.randn(self.batch_size, self.max_n_al)
        self.y_cucu = np.random.randn(self.batch_size, self.max_n_cu)
        self.y_cual = np.random.randn(self.batch_size, self.max_n_cu)

        with tf.name_scope("Inputs"):
            self.descriptors = AttributeDict(
                Al=(tf.convert_to_tensor(self.g_al, tf.float64, 'g_al'),
                    tf.convert_to_tensor(self.m_al, tf.float64, 'm_al')),
                Cu=(tf.convert_to_tensor(self.g_cu, tf.float64, 'g_cu'),
                    tf.convert_to_tensor(self.m_cu, tf.float64, 'm_cu')))
            self.positions = tf.constant(
                np.random.rand(1, self.max_n_atoms, 3),
                dtype=tf.float64,
                name='positions')
            self.mask = np.ones((self.batch_size, self.max_n_atoms), np.float64)
            self.features = AttributeDict(
                descriptors=self.descriptors, positions=self.positions,
                mask=self.mask)
            self.atomic_splits = AttributeDict(
                AlAl=tf.convert_to_tensor(self.y_alal, tf.float64, 'y_alal'),
                AlCu=tf.convert_to_tensor(self.y_alcu, tf.float64, 'y_alcu'),
                CuCu=tf.convert_to_tensor(self.y_cucu, tf.float64, 'y_cucu'),
                CuAl=tf.convert_to_tensor(self.y_cual, tf.float64, 'y_cual'))
            self.symmetric_atomic_splits = AttributeDict(
                AlAl=tf.convert_to_tensor(self.y_alal, tf.float64, 'y_alal'),
                CuCu=tf.convert_to_tensor(self.y_cucu, tf.float64, 'y_cucu'),
                AlCu=tf.convert_to_tensor(
                    np.concatenate((self.y_alcu, self.y_cual), axis=1),
                    tf.float64, 'y_alcu'))


def test_inference():
    """
    Test the inference: `EamNN.build`.
    """
    with tf.Graph().as_default():

        data = TestData()

        nn = EamNN(elements=data.elements, hidden_sizes=8, symmetric=False,
                   forces=False)

        predictions = nn.build(data.features, verbose=True)
        assert_list_equal(
            predictions.energy.shape.as_list(), [data.batch_size, ])


def test_symmetric_inference():
    """
    Test the inference of the symmetric nn-EAM.
    """
    with tf.Graph().as_default():

        data = TestData()

        nn = EamNN(elements=data.elements, hidden_sizes=8, symmetric=True,
                   forces=False)

        predictions = nn.build(data.features, verbose=True)
        assert_list_equal(
            predictions.energy.shape.as_list(), [data.batch_size, ])


def test_dynamic_stitch():
    """
    Test the method `EamNN._dynamic_stitch`.
    """
    with tf.Graph().as_default():

        data = TestData()

        nn = EamNN(elements=data.elements, symmetric=False, forces=False)
        symm_nn = EamNN(elements=data.elements, symmetric=True, forces=False)

        op = nn._dynamic_stitch(data.atomic_splits)
        symm_op = symm_nn._dynamic_stitch(data.symmetric_atomic_splits)

        ref = np.concatenate((data.y_alal + data.y_alcu,
                              data.y_cucu + data.y_cual),
                             axis=1)

        with tf.Session() as sess:
            results = sess.run([op, symm_op])

            assert_tuple_equal(results[0].shape, ref.shape)
            assert_tuple_equal(results[1].shape, ref.shape)

            assert_array_equal(results[0], ref)
            assert_array_equal(results[1], ref)


def test_dynamic_partition():
    """
    Test the method `EamNN._dynamic_partition`.
    """
    with tf.Graph().as_default():

        data = TestData()

        nn = EamNN(elements=data.elements, symmetric=False, forces=False)
        symm_nn = EamNN(elements=data.elements, symmetric=True, forces=False)

        with tf.Session() as sess:
            results = sess.run(nn._dynamic_partition(data.features))

            assert_equal(len(results), 4)
            assert_array_equal(results['AlAl'][0], data.g_al[:, [0]])
            assert_array_equal(results['AlCu'][0], data.g_al[:, [1]])
            assert_array_equal(results['CuCu'][0], data.g_cu[:, [0]])
            assert_array_equal(results['CuAl'][0], data.g_cu[:, [1]])

            assert_array_equal(results['AlAl'][1], data.m_al[:, [0]])
            assert_array_equal(results['AlCu'][1], data.m_al[:, [1]])
            assert_array_equal(results['CuCu'][1], data.m_cu[:, [0]])
            assert_array_equal(results['CuAl'][1], data.m_cu[:, [1]])

            results = sess.run(symm_nn._dynamic_partition(data.features))

            assert_equal(len(results), 3)
            assert_array_equal(results['AlAl'][0], data.g_al[:, [0]])
            assert_array_equal(results['CuCu'][0], data.g_cu[:, [0]])

            assert_array_equal(results['AlAl'][1], data.m_al[:, [0]])
            assert_array_equal(results['CuCu'][1], data.m_cu[:, [0]])

            assert_array_equal(results['AlCu'][0],
                               np.concatenate((data.g_al[:, [1]],
                                               data.g_cu[:, [1]]), 2))
            assert_array_equal(results['AlCu'][1],
                               np.concatenate((data.m_al[:, [1]],
                                               data.m_cu[:, [1]]), 2))


if __name__ == "__main__":
    nose.run()
