# coding=utf-8
"""
This module defines unit tests of `EamNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_equal

from ..eam import EamNN
from tensoralloy.misc import AttributeDict
from tensoralloy.test_utils import assert_array_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_dynamic_partition():
    """
    Test the method `EamNN._dynamic_partition`.
    """
    with tf.Graph().as_default():

        batch_size = 10
        max_n_terms = 2
        max_n_al = 5
        max_n_cu = 7
        nnl = 10
        elements = sorted(['Cu', 'Al'])

        shape_al = (batch_size, max_n_terms, max_n_al, nnl)
        shape_cu = (batch_size, max_n_terms, max_n_cu, nnl)

        g_al = np.random.randn(*shape_al)
        m_al = np.random.randint(0, 2, shape_al).astype(np.float64)
        g_cu = np.random.randn(*shape_cu)
        m_cu = np.random.randint(0, 2, shape_cu).astype(np.float64)

        descriptors = AttributeDict(
            Al=(tf.convert_to_tensor(g_al, dtype=tf.float64, name='g_al'),
                tf.convert_to_tensor(m_al, dtype=tf.float64, name='m_al')),
            Cu=(tf.convert_to_tensor(g_cu, dtype=tf.float64, name='g_cu'),
                tf.convert_to_tensor(m_cu, dtype=tf.float64, name='m_cu'))
        )
        features = AttributeDict(descriptors=descriptors)

        nn = EamNN(elements=elements, symmetric=False)
        symm_nn = EamNN(elements=elements, symmetric=True)

        with tf.Session() as sess:
            results = sess.run(nn._dynamic_partition(features))

            assert_equal(len(results), 4)
            assert_array_equal(results['AlAl'][0], g_al[:, [0]])
            assert_array_equal(results['AlCu'][0], g_al[:, [1]])
            assert_array_equal(results['CuCu'][0], g_cu[:, [0]])
            assert_array_equal(results['CuAl'][0], g_cu[:, [1]])

            assert_array_equal(results['AlAl'][1], m_al[:, [0]])
            assert_array_equal(results['AlCu'][1], m_al[:, [1]])
            assert_array_equal(results['CuCu'][1], m_cu[:, [0]])
            assert_array_equal(results['CuAl'][1], m_cu[:, [1]])

            results = sess.run(symm_nn._dynamic_partition(features))

            assert_equal(len(results), 3)
            assert_array_equal(results['AlAl'][0], g_al[:, [0]])
            assert_array_equal(results['CuCu'][0], g_cu[:, [0]])

            assert_array_equal(results['AlAl'][1], m_al[:, [0]])
            assert_array_equal(results['CuCu'][1], m_cu[:, [0]])

            assert_array_equal(results['AlCu'][0],
                               np.concatenate((g_al[:, [1]], g_cu[:, [1]]), 2))
            assert_array_equal(results['AlCu'][1],
                               np.concatenate((m_al[:, [1]], m_cu[:, [1]]), 2))



if __name__ == "__main__":
    nose.run()
