# coding=utf-8
"""
This module defines unit tests of the module `losses`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_less, assert_equal
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensoralloy.nn.dataclasses import ForcesLossOptions, EnergyLossOptions
from tensoralloy.nn.dataclasses import StressLossOptions
from tensoralloy.nn.losses import get_energy_loss, get_forces_loss
from tensoralloy.nn.losses import get_stress_loss, adaptive_sample_weight
from tensoralloy.precision import precision_scope, get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_energy_loss():
    """
    Test the function `get_energy_loss`.
    """
    with precision_scope('medium'):

        dtype = get_float_dtype()

        x = np.random.uniform(0.0, 3.0, size=(6, )).astype(dtype.as_numpy_dtype)
        y = np.random.uniform(0.0, 3.0, size=(6, )).astype(dtype.as_numpy_dtype)
        n = np.random.randint(1, 5, size=(6, )).astype(dtype.as_numpy_dtype)

        y_rmse = np.sqrt(mean_squared_error(x / n, y / n))
        y_mae = mean_absolute_error(x / n, y / n)

        with tf.Graph().as_default():

            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            n = tf.convert_to_tensor(n)

            rmse = get_energy_loss(x, y, n_atoms=n, collections=['UnitTest'])
            assert_equal(len(tf.get_collection('UnitTest')), 3)

            mae = tf.get_default_graph().get_tensor_by_name('Energy/mae:0')

            with tf.Session() as sess:
                assert_less(y_rmse - sess.run(rmse), 1e-8)
                assert_less(y_mae - sess.run(mae), 1e-8)


def test_weighted_energy_loss():
    """
    Test the function `get_energy_loss` with sample weights.
    """
    with precision_scope('high'):

        options = EnergyLossOptions(per_atom_loss=True)
        dtype = get_float_dtype()

        x = np.random.uniform(0.0, 3.0, size=(6, ))
        y = np.random.uniform(0.0, 3.0, size=(6, ))
        n = np.random.randint(1, 5, size=(6, ))
        w = np.random.uniform(0.0, 1.0, size=(6, ))

        y_rmse = np.sqrt(((x / n - y / n)**2 * w / w.sum()).sum())       
        y_mae = mean_absolute_error(x / n, y / n)

        with tf.Graph().as_default():

            x = tf.convert_to_tensor(x, dtype=dtype)
            y = tf.convert_to_tensor(y, dtype=dtype)
            n = tf.convert_to_tensor(n, dtype=dtype)
            w = tf.convert_to_tensor(w, dtype=dtype)

            rmse = get_energy_loss(
                x, y, n_atoms=n, collections=['UnitTest'], options=options,
                sample_weight=w, normalized_weight=True)
            assert_equal(len(tf.get_collection('UnitTest')), 3)

            mae = tf.get_default_graph().get_tensor_by_name('Energy/mae/atom:0')

            with tf.Session() as sess:
                assert_less(y_rmse - sess.run(rmse), 1e-8)
                assert_less(y_mae - sess.run(mae), 1e-8)


def test_forces_loss():
    """
    Test the function `get_forces_loss`.
    """
    with precision_scope('high'):

        x = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
        y = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
        n_atoms = np.asarray([5, 6, 4, 8, 5, 3])
        mask = np.zeros((6, 9))
        for i in range(len(n_atoms)):
            mask[i, 1: n_atoms[i] + 1] = 1.0
        x[5] = 0.0
        y[5] = 0.0

        mae_values = []
        rmse_values = []

        for i in range(len(x)):
            for j in range(n_atoms[i]):
                mae_values.append(np.abs(x[i, j] - y[i, j]))
                rmse_values.append(np.square(x[i, j] - y[i, j]))

        y_mae = np.mean(mae_values)
        y_rmse = np.sqrt(np.mean(rmse_values))

        with tf.Graph().as_default():

            x = tf.convert_to_tensor(np.insert(x, 0, 0, axis=1))
            y = tf.convert_to_tensor(y)
            mask = tf.convert_to_tensor(mask)
            options = ForcesLossOptions(weight=10.0)

            rmse = get_forces_loss(x, y, atom_masks=mask, options=options,
                                   collections=['UnitTest'])
            mae = tf.get_default_graph().get_tensor_by_name(
                'Forces/Absolute/mae:0')

            with tf.Session() as sess:
                assert_less(y_mae - sess.run(mae), 1e-8)
                assert_less(y_rmse * 10.0 - sess.run(rmse), 1e-8)


def test_stress_loss():
    """
    Test the function `get_stress_loss`.
    """
    with precision_scope('high'):

        x = np.random.uniform(0.1, 3.0, size=(8, 6))
        y = np.random.uniform(0.1, 3.0, size=(8, 6))
        y_rmse = np.mean(
            np.linalg.norm(x - y, axis=1) / np.linalg.norm(x, axis=1))

        with tf.Graph().as_default():
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)

            rmse = get_stress_loss(x, y)

            with tf.Session() as sess:
                assert_less(y_rmse - sess.run(rmse), 1e-8)


def test_weighted_stress_loss():
    """
    Test the function `get_stress_loss` with sample weights.
    """
    with precision_scope('high'):

        x = np.random.uniform(0.1, 3.0, size=(8, 6))
        y = np.random.uniform(0.1, 3.0, size=(8, 6))
        w = np.random.uniform(0.0, 1.0, size=(8, ))
        y_rmse = np.sqrt(mean_squared_error(x, y, sample_weight=w))

        dtype = get_float_dtype()

        with tf.Graph().as_default():
            x = tf.convert_to_tensor(x, dtype=dtype)
            y = tf.convert_to_tensor(y, dtype=dtype)
            w = tf.convert_to_tensor(w, dtype=dtype)

            rmse = get_stress_loss(x, y, sample_weight=w, 
                                   normalized_weight=True)

            with tf.Session() as sess:
                print(sess.run(rmse))
                print(y_rmse)
                assert_less(y_rmse - sess.run(rmse), 1e-8)


def sigmoid(x, a, b, c, d):
    return c / (1 + np.exp(-a * (b - x))) + d


def test_adaptive_sample_weight():
    """
    Test the function `adaptive_sample_weight`.
    """
    with precision_scope('high'):
        np.random.seed(1)
        a = 1
        b = 2
        c = 2
        d = 0.1
        forces = np.random.uniform(0.0, 3.0, size=(6, 8, 3))
        natoms = np.asarray([5, 6, 4, 8, 5, 3])
        mask = np.zeros((6, 9))
        for i in range(len(natoms)):
            mask[i, 1: natoms[i] + 1] = 1.0
        forces[5] = 0.0
        ref = np.zeros(len(forces))
        for i in range(len(forces)):
            f = np.sqrt((forces[i]**2).sum() / natoms[i])
            ref[i] = sigmoid(f, a, b, c, d)
        
        with tf.Graph().as_default():
            x_ = tf.convert_to_tensor(forces)
            mask_ = tf.convert_to_tensor(mask)
            natoms_ = tf.convert_to_tensor(
                natoms, dtype=tf.float64, name="natoms")

            w = adaptive_sample_weight(x_, mask_, natoms_, a, b, c, d)
            with tf.Session() as sess:
                print(sess.run(w))
                print(ref)


if __name__ == "__main__":
    test_weighted_stress_loss()
