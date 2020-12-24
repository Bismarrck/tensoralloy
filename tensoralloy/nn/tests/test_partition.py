#!coding=utf-8
"""
The unit tests of module `partition`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_equal
from collections import Counter

from tensoralloy.utils import get_kbody_terms, ModeKeys
from tensoralloy.test_utils import assert_array_equal
from tensoralloy.nn.partition import dynamic_partition

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AlCuFakeData:
    """
    A fake dataset for unit tests in this module.
    """

    def __init__(self):
        """
        Initialization method.
        """
        self.batch_size = 10
        self.max_n_terms = 2
        self.max_n_al = 5
        self.max_n_cu = 7
        self.max_n_atoms = self.max_n_al + self.max_n_cu
        self.max_occurs = Counter({'Al': self.max_n_al, 'Cu': self.max_n_cu})
        self.nnl = 10
        self.elements = sorted(['Cu', 'Al'])
        self.kbody_terms_for_element = get_kbody_terms(self.elements)[1]

        shape_al = (4,
                    self.batch_size,
                    self.max_n_terms,
                    self.max_n_al,
                    self.nnl,
                    1)
        shape_cu = (4,
                    self.batch_size,
                    self.max_n_terms,
                    self.max_n_cu,
                    self.nnl,
                    1)

        self.g_al = np.random.randn(*shape_al)
        self.g_cu = np.random.randn(*shape_cu)

        # The shape of `mask` does not expand.
        self.m_al = np.random.randint(0, 2, shape_al[1:]).astype(np.float64)
        self.m_cu = np.random.randint(0, 2, shape_cu[1:]).astype(np.float64)
        self.y_alal = np.random.randn(self.batch_size, self.max_n_al)
        self.y_alcu = np.random.randn(self.batch_size, self.max_n_al)
        self.y_cucu = np.random.randn(self.batch_size, self.max_n_cu)
        self.y_cual = np.random.randn(self.batch_size, self.max_n_cu)

        with tf.name_scope("Inputs"):
            self.dists_and_masks = dict(
                Al=(tf.convert_to_tensor(self.g_al, tf.float64, 'g_al'),
                    tf.convert_to_tensor(self.m_al, tf.float64, 'm_al')),
                Cu=(tf.convert_to_tensor(self.g_cu, tf.float64, 'g_cu'),
                    tf.convert_to_tensor(self.m_cu, tf.float64, 'm_cu')))


def test_dynamic_partition_radial():
    """
    Test the method `EamNN._dynamic_partition`.
    """
    with tf.Graph().as_default():
        data = AlCuFakeData()
        mode = ModeKeys.TRAIN

        with tf.Session() as sess:
            op, max_occurs = dynamic_partition(
                data.dists_and_masks,
                elements=data.elements,
                kbody_terms_for_element=data.kbody_terms_for_element,
                mode=mode,
                merge_symmetric=False)
            results = sess.run(op)

            assert_equal(len(max_occurs), 2)
            assert_equal(max_occurs['Al'], data.max_n_al)
            assert_equal(max_occurs['Cu'], data.max_n_cu)
            assert_equal(len(results), 4)
            assert_array_equal(results['AlAl'][0][0], data.g_al[0][:, [0]])
            assert_array_equal(results['AlCu'][0][0], data.g_al[0][:, [1]])
            assert_array_equal(results['CuCu'][0][0], data.g_cu[0][:, [0]])
            assert_array_equal(results['CuAl'][0][0], data.g_cu[0][:, [1]])

            assert_array_equal(results['AlAl'][1], data.m_al[:, [0]])
            assert_array_equal(results['AlCu'][1], data.m_al[:, [1]])
            assert_array_equal(results['CuCu'][1], data.m_cu[:, [0]])
            assert_array_equal(results['CuAl'][1], data.m_cu[:, [1]])

            op = dynamic_partition(
                data.dists_and_masks,
                elements=data.elements,
                kbody_terms_for_element=data.kbody_terms_for_element,
                mode=mode,
                merge_symmetric=True)[0]
            results = sess.run(op)

            assert_equal(len(results), 3)
            assert_array_equal(results['AlAl'][0][0], data.g_al[0][:, [0]])
            assert_array_equal(results['CuCu'][0][0], data.g_cu[0][:, [0]])
            assert_array_equal(results['AlCu'][0][0],
                               np.concatenate((data.g_al[0][:, [1]],
                                               data.g_cu[0][:, [1]]), 2))
            assert_array_equal(results['AlAl'][1], data.m_al[:, [0]])
            assert_array_equal(results['CuCu'][1], data.m_cu[:, [0]])
            assert_array_equal(results['AlCu'][1],
                               np.concatenate((data.m_al[:, [1]],
                                               data.m_cu[:, [1]]), 2))


def test_dynamic_partition_angular():
    pass


if __name__ == "__main__":
    nose.main()
