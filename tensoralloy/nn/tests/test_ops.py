# coding=utf-8
"""
This module defines tests of module `tensoralloy.nn.ops`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose

from ase.db import connect
from os.path import join
from nose.tools import assert_equal

from tensoralloy.utils import AttributeDict
from tensoralloy.test_utils import test_dir
from tensoralloy.dataset import Dataset
from tensoralloy.nn.ops import get_train_op
from tensoralloy.nn.atomic import AtomicNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_train_op(trainable=False):
    """
    A helper function to created a `train_op`.
    """
    work_dir = join(test_dir(), 'Ni')
    dataset = Dataset(connect(join(work_dir, 'Ni.db')), name='Ni',
                      descriptor='behler', serial=True, trainable=trainable)

    if not dataset.load_tfrecords(work_dir):
        dataset.to_records(work_dir, test_size=1)
    assert_equal(dataset.test_size, 1)

    input_fn = dataset.input_fn(mode=tf.estimator.ModeKeys.TRAIN,
                                batch_size=1,
                                num_epochs=None,
                                shuffle=False)

    features, labels = input_fn()

    hparams = AttributeDict(
        loss=AttributeDict(
            energy=AttributeDict(weight=1.0)),
        opt=AttributeDict(
            learning_rate=0.01, decay_function=None, decay_steps=1000,
            decay_rate=0.9, staircase=False, method='adam'))

    nn = AtomicNN(elements=dataset.transformer.elements)
    nn.attach_transformer(dataset.transformer)

    predictions = nn.build(features, tf.estimator.ModeKeys.TRAIN)
    total_loss, losses = nn.get_total_loss(
        predictions, labels, features.n_atoms, hparams)

    return get_train_op(losses, hparams, nn.minimize_properties)


def test_get_train_op():
    """
    Test the function `get_train_op`.
    """
    tf.reset_default_graph()

    graph = tf.Graph()

    with graph.as_default():

        _get_train_op(trainable=False)

        assert_equal(len(tf.trainable_variables()), 5)
        assert_equal(len(tf.moving_average_variables()), 5)
        assert_equal(len(tf.model_variables()), 9)

    graph = tf.Graph()

    with graph.as_default():
        _get_train_op(trainable=True)

        assert_equal(len(tf.trainable_variables()), 9)
        assert_equal(len(tf.moving_average_variables()), 9)
        assert_equal(len(tf.model_variables()), 9)

        # varialbles 9, moving averged 9, 9 * 2
        # global step 1
        # adam 2
        # 2 adam variables per variable, 9 * 2
        assert_equal(len(tf.global_variables()), 9 * 2 + 1 + 2 + 9 * 2)


if __name__ == "__main__":
    nose.main()
