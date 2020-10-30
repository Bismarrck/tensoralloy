# coding=utf-8
"""
This module defines tests of module `tensoralloy.nn.ops`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose
import glob

from os import remove
from os.path import join, exists
from nose.tools import assert_equal, with_setup
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.test_utils import test_dir
from tensoralloy.train.dataset import Dataset
from tensoralloy.io.db import connect
from tensoralloy.io.read import read_file
from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.nn.opt import get_train_op
from tensoralloy.nn.atomic import SymmetryFunction, AtomicNN
from tensoralloy.nn.dataclasses import OptParameters, LossParameters

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def run_case():
    """
    A helper function to created a `train_op`.
    """
    work_dir = join(test_dir(), "datasets", "Ni")

    db_file = join(work_dir, "Ni.db")
    if not exists(db_file):
        database = read_file(join(work_dir, "Ni.extxyz"), verbose=False)
    else:
        database = connect(db_file)
    rc = 6.0
    clf = BatchUniversalTransformer(
        rcut=rc,
        max_occurs=database.max_occurs,
        nij_max=database.get_nij_max(rc, allow_calculation=True),
        nnl_max=database.get_nnl_max(rc, allow_calculation=True),
        angular=False)

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

    elements = dataset.transformer.elements
    sf = SymmetryFunction(elements=elements)
    nn = AtomicNN(elements=elements, descriptor=sf, minmax_scale=False)
    nn.attach_transformer(dataset.transformer)

    predictions = nn.build(features, tf_estimator.ModeKeys.TRAIN)
    total_loss, losses = nn.get_total_loss(
        predictions=predictions,
        labels=labels,
        n_atoms=features["n_atoms_vap"],
        atom_masks=features["atom_masks"],
        loss_parameters=loss_parameters)

    return get_train_op(losses, opt_parameters, nn.minimize_properties)


def cleanup():
    """
    The cleanup function.
    """
    data_dir = join(test_dir(), "datasets", "Ni")
    db_file = join(test_dir(), "datasets", "Ni", "Ni.db")
    if exists(db_file):
        remove(db_file)
    for tfrecord_file in glob.glob(f"{data_dir}/*.tfrecords"):
        remove(tfrecord_file)


@with_setup(teardown=cleanup)
def test_get_train_op():
    """
    Test the function `get_train_op`.
    """
    with tf.Graph().as_default():

        tf.train.get_or_create_global_step()
        run_case()

        # eta0, eta1, eta2, eta3, omega0
        # xlo, xhi
        # kernel 1,2,3 (trainable)
        # bias 1,2 (trainable)
        # bias 3 (trainable, output)

        assert_equal(len(tf.trainable_variables()), 6)
        assert_equal(len(tf.moving_average_variables()), 6)
        print(tf.model_variables())
        assert_equal(len(tf.model_variables()), 6)


if __name__ == "__main__":
    nose.main()
