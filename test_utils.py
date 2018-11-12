# coding=utf-8
"""
This module defines unit tests for functions in `utils`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import utils
from nose import main
from nose.tools import assert_less

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_batch_gather_positions():
    with tf.Graph().as_default():
        R = np.arange(18.0, dtype=np.float64).reshape((2, 3, 3))
        indices = np.array([[0, 0, 1, 1], [1, 2, 2, 1]])
        results = np.zeros((2, 4, 3), dtype=np.float64)
        for i, row in enumerate(indices):
            results[i] = R[i][row]
        g = utils.batch_gather_positions(R, indices)
        with tf.Session() as sess:
            values = sess.run(g)
        assert_less(np.abs(values - results).max(), 1e-8)


if __name__ == "__main__":
    main()
