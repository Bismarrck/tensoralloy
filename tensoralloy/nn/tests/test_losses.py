# coding=utf-8
"""
This module defines unit tests of the module `losses`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_less
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..losses import norm_loss, rmse_loss, mae_loss

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_norm_loss():
    """
    Test the function `norm_loss`.
    """
    x = np.random.rand(6, 8, 3)
    x[5, 7] = 0.0
    y = np.random.rand(6, 8, 3)

    values = []
    for i in range(6):
        for j in range(8):
            a = np.linalg.norm(x[i, j])
            if a == 0.0:
                values.append(0.0)
            else:
                b = np.linalg.norm(x[i, j] - y[i, j])
                values.append(b / a)
    ref = np.mean(values)

    with tf.Graph().as_default():
        loss = norm_loss(x, y, name='loss')
        with tf.Session() as sess:
            assert_less(ref - sess.run(loss), 1e-8)


def test_rmse_and_mae_losses():
    """
    Test the functions `rmse_loss` and `mae_loss`.
    """
    x = np.random.rand(6, 8, 3)
    x[5] = 0.0
    y = np.random.rand(6, 8, 3)
    y[5] = 0.0

    mae_values = []
    rmse_values = []

    for i in range(len(x)):
        mae_values.append(mean_absolute_error(x[i], y[i]))
        rmse_values.append(mean_squared_error(x[i], y[i]))

    y_mae = np.mean(mae_values)
    y_rmse = np.mean(rmse_values)

    with tf.Graph().as_default():

        mae = mae_loss(x, y, name='mae')
        rmse = rmse_loss(x, y, name='rmse')

        with tf.Session() as sess:
            assert_less(y_mae - sess.run(mae), 1e-8)
            assert_less(y_rmse - sess.run(rmse), 1e-8)


if __name__ == "__main__":
    nose.run()
