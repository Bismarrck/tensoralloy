#coding=utf-8
"""
Unit tests of `tensoralloy.nn.utils`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from tensoralloy.precision import precision_scope
from tensoralloy.nn.utils import softplus
from tensoralloy.test_utils import assert_array_almost_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_softplus():
    with tf.Graph().as_default():
        with precision_scope("high"):
            x = np.linspace(-10, 50, num=61, endpoint=True).astype(np.float64)
            with tf.Session() as sess:
                y, z = sess.run([tf.nn.softplus(x), softplus(x, limit=30)])
                assert_array_almost_equal(y, z, delta=1e-8)


if __name__ == "__main__":
    nose.main()
