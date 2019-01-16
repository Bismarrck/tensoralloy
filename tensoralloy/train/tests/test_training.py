# coding=utf-8
"""
This module defines tests of `TrainingManager`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import shutil

from os.path import join, exists
from nose.tools import assert_equal, assert_less, assert_is_none
from nose.tools import with_setup

from tensoralloy.train.training import TrainingManager
from tensoralloy.misc import test_dir
from tensoralloy.transformer import BatchSymmetryFunctionTransformer
from tensoralloy.descriptor.tests.test_cutoff import polynomial_cutoff_simple
from tensoralloy.dtypes import get_float_dtype, set_float_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def teardown_initialization():
    """
    The cleanup function for `test_initialization`.
    """
    model_dir = join(test_dir(), 'inputs', 'model')
    if exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)


@with_setup(teardown=teardown_initialization)
def test_initialization():
    """
    Test the initialization of a `TrainingManager`.
    """
    input_file = join(test_dir(), 'inputs', 'Ni.behler.k2.toml')
    manager = TrainingManager(input_file)
    transformer = manager.dataset.transformer
    hparams = manager.hparams

    assert_equal(get_float_dtype(), tf.float32)

    assert_equal(manager.dataset.descriptor, 'behler')
    assert_equal(manager.dataset.test_size, 1)
    assert_equal(manager.dataset.cutoff_radius, 6.0)

    assert isinstance(transformer, BatchSymmetryFunctionTransformer)
    assert_equal(transformer.trainable, True)

    with tf.Session() as sess:
        rc = transformer.rc
        gamma = 5.0
        r = np.linspace(1.0, 10.0, num=91, endpoint=True)
        x = np.asarray([polynomial_cutoff_simple(ri, rc, gamma) for ri in r])
        y = sess.run(
            transformer._cutoff_fn(
                tf.convert_to_tensor(r, dtype=tf.float64),
                name='cutoff'))

        assert_less(np.abs(x - y).max(), 1e-8)

    assert_equal(hparams.opt.method, 'adam')
    assert_equal(hparams.opt.learning_rate, 0.01)
    assert_is_none(hparams.opt.decay_function)
    assert_equal(hparams.precision, 'medium')
    assert_equal(hparams.seed, 1958)

    assert_equal(manager.nn.positive_energy_mode, True)

    set_float_precision()



if __name__ == "__main__":
    nose.main()
