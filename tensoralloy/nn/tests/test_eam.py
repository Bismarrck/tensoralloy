# coding=utf-8
"""
This module defines unit tests of `EamNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_less, assert_equal

from ..eam import EamNN
from tensoralloy.misc import AttributeDict

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
        eps = 1e-8
        elements = sorted(['Cu', 'Al'])

        g_al = np.random.randn(batch_size, max_n_terms, max_n_al, nnl)
        g_cu = np.random.randn(batch_size, max_n_terms, max_n_cu, nnl)

        descriptors = AttributeDict(
            Al=tf.convert_to_tensor(g_al, dtype=tf.float64, name='g_al'),
            Cu=tf.convert_to_tensor(g_cu, dtype=tf.float64, name='g_cu')
        )
        features = AttributeDict(descriptors=descriptors)

        nn = EamNN(elements=elements, symmetric=False)
        symm_nn = EamNN(elements=elements, symmetric=True)

        with tf.Session() as sess:
            results = sess.run(nn._dynamic_partition(features))

            assert_equal(len(results), 4)
            assert_less(np.abs(results['AlAl'] - g_al[:, [0]]).max(), eps)
            assert_less(np.abs(results['AlCu'] - g_al[:, [1]]).max(), eps)
            assert_less(np.abs(results['CuCu'] - g_cu[:, [0]]).max(), eps)
            assert_less(np.abs(results['CuAl'] - g_cu[:, [1]]).max(), eps)

            results = sess.run(symm_nn._dynamic_partition(features))

            assert_equal(len(results), 3)
            assert_less(np.abs(results['AlAl'] - g_al[:, [0]]).max(), eps)
            assert_less(np.abs(results['CuCu'] - g_cu[:, [0]]).max(), eps)

            stack = np.concatenate((g_al[:, [1]], g_cu[:, [1]]), axis=2)
            assert_less(np.abs(results['AlCu'] - stack).max(), eps)


if __name__ == "__main__":
    nose.run()
