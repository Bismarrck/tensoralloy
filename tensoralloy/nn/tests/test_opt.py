# coding=utf-8
"""
This module defines tests of module `tensoralloy.nn.ops`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose

from os.path import join
from nose.tools import assert_equal
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.test_utils import test_dir
from tensoralloy.dataset import Dataset
from tensoralloy.io.db import connect
from tensoralloy.transformer import BatchSymmetryFunctionTransformer
from tensoralloy.nn.opt import get_train_op
from tensoralloy.nn.atomic import AtomicNN
from tensoralloy.nn.dataclasses import OptParameters, LossParameters
from tensoralloy.utils import Defaults

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _get_train_op(trainable=False):
    """
    A helper function to created a `train_op`.
    """
    work_dir = join(test_dir(), 'Ni')

    database = connect(join(work_dir, 'Ni.db'))
    clf = BatchSymmetryFunctionTransformer(
        rc=Defaults.rc,
        max_occurs=database.max_occurs,
        nij_max=database.get_nij_max(Defaults.rc),
        nijk_max=0,
        angular=False,
        trainable=trainable)

    dataset = Dataset(database, name='Ni', transformer=clf, serial=True)

    if not dataset.load_tfrecords(work_dir):
        dataset.to_records(work_dir, test_size=1)
    assert_equal(dataset.test_size, 1)

    input_fn = dataset.input_fn(mode=tf_estimator.ModeKeys.TRAIN,
                                batch_size=1,
                                num_epochs=None,
                                shuffle=False)

    features, labels = input_fn()

    loss_parameters = LossParameters()
    loss_parameters.energy.weight = 1.0

    opt_parameters = OptParameters(learning_rate=0.01, decay_function=None,
                                   decay_rate=0.9, staircase=False,
                                   method='Adam')

    nn = AtomicNN(elements=dataset.transformer.elements)
    nn.attach_transformer(dataset.transformer)

    predictions = nn.build(features, tf_estimator.ModeKeys.TRAIN)
    total_loss, losses = nn.get_total_loss(
        predictions, labels, features.n_atoms, loss_parameters=loss_parameters)

    return get_train_op(losses, opt_parameters, nn.minimize_properties)


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
        assert_equal(len(tf.model_variables()), 10)

    graph = tf.Graph()

    with graph.as_default():
        _get_train_op(trainable=True)

        assert_equal(len(tf.trainable_variables()), 10)
        assert_equal(len(tf.moving_average_variables()), 10)
        assert_equal(len(tf.model_variables()), 10)

        # varialbles 10, moving averged 10, 10 * 2
        # global step 1
        # adam 2
        # 2 adam variables per variable, 10 * 2
        assert_equal(len(tf.global_variables()), 10 * 2 + 1 + 2 + 10 * 2)


if __name__ == "__main__":
    nose.main()
