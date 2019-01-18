# coding=utf-8
"""
This module defines tests of `TrainingManager`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import shutil
import glob
import os

from unittest import skipUnless
from os.path import join, exists
from nose.tools import assert_equal, assert_less, assert_is_none, assert_in
from nose.tools import with_setup, assert_dict_equal

from tensoralloy.utils import Defaults
from tensoralloy.train.training import TrainingManager
from tensoralloy.nn import EamAlloyNN
from tensoralloy.test_utils import test_dir, assert_array_almost_equal
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


def teardown_initialize_eam_training():
    """
    The cleanup function for `test_initialize_eam_training`.
    """
    model_dir = join(test_dir(), 'inputs', 'snap_Ni_zjw04')
    if exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)

    tfrecords_dir = join(test_dir(), 'inputs', 'temp')
    if exists(tfrecords_dir):
        shutil.rmtree(tfrecords_dir, ignore_errors=True)


@skipUnless(os.environ.get('TEST_EXPERIMENTS'),
            "The flag 'TEST_EXPERIMENTS' is not set")
@with_setup(teardown=teardown_initialize_eam_training)
def test_initialize_eam_training():
    """
    Test initializing an EAM training experiment.
    """
    input_file = join(test_dir(), 'inputs', 'snap_Ni.zjw04.toml')
    manager = TrainingManager(input_file)

    assert_equal(manager.dataset.cutoff_radius, 6.0)

    nn = manager.nn
    assert isinstance(nn, EamAlloyNN)

    assert_dict_equal(nn.potentials, {"Ni": {"rho": "zjw04", "embed": "zjw04"},
                                      "NiNi": {"phi": "zjw04"}})
    assert_equal(nn.positive_energy_mode, False)

    manager.train_and_evaluate()


class DebugWarmStartFromVariablesHook(tf.train.SessionRunHook):
    """
    This hook can be used to replace `tf.estimator.WarmStartSettings`.
    """

    def __init__(self,
                 previous_checkpoint: str,
                 true_values: dict):
        """
        Initialization method.

        Parameters
        ----------
        previous_checkpoint : str
            The previous checkpoint to load.

        """
        tf.logging.info("Create WarmStartFromVariablesHook.")

        self._previous_checkpoint = previous_checkpoint
        self._saver = None
        self._ema = None
        self._true_values = true_values

    def begin(self):
        """
        Create restoring operations before the graph been finalized.
        """
        tf.logging.info('Initialize a Saver to restore EMA variables.')
        self._ema = tf.train.ExponentialMovingAverage(
            Defaults.variable_moving_average_decay)
        self._saver = tf.train.Saver(
            var_list=self._ema.variables_to_restore(tf.trainable_variables()))

    def after_create_session(self,
                             session: tf.Session,
                             coord: tf.train.Coordinator):
        """
        When this is called, the graph is finalized and ops can no longer be
        added to the graph.
        """
        tf.logging.info(
            f'Restore EMA variables from {self._previous_checkpoint}')
        self._saver.restore(session, self._previous_checkpoint)

        values = session.run({var.name: var for var in tf.model_variables()})
        for var_name in self._true_values.keys():
            assert_in(var_name, values)
            assert_array_almost_equal(self._true_values[var_name],
                                      values[var_name], delta=1e-10)


def teardown_warm_start():
    """
    The cleanup function for `test_warm_start`.
    """
    top_dir = join(test_dir(), 'inputs', 'warm_start')
    first_dir = join(top_dir, 'first')
    second_dir = join(top_dir, 'second')

    if exists(first_dir):
        shutil.rmtree(first_dir)

    if exists(second_dir):
        shutil.rmtree(second_dir)

    files = list(glob.glob(join(top_dir, "*.behler.tfrecords")))
    for afile in files:
        os.remove(afile)


@skipUnless(os.environ.get('TEST_EXPERIMENTS'),
            "The flag 'TEST_EXPERIMENTS' is not set")
@with_setup(teardown=teardown_warm_start)
def test_warm_start():
    """
    Test fine-tuning a model by reading moving averaged variables of a previous
    checkpoint.
    """
    top_dir = join(test_dir(), 'inputs', 'warm_start')

    # Initialize the first training manager and train 100 steps.
    first = TrainingManager(join(top_dir, 'first.toml'))
    first.train_and_evaluate()

    # Explicitly read the lates checkpoint file and obtain the moving averaged
    # variable weights.
    with tf.Graph().as_default():
        saver = tf.train.import_meta_graph(
            join(top_dir, 'first', 'model.ckpt-100.meta'))
        variable_ops = {}
        for var in tf.global_variables():
            if var.op.name.endswith("ExponentialMovingAverage"):
                name = var.op.name.replace('/ExponentialMovingAverage', '')
                variable_ops[f"{name}:0"] = var
        with tf.Session() as sess:
            saver.restore(sess, join(top_dir, 'first', 'model.ckpt-100'))
            values = sess.run(variable_ops)

    # Set the default graph to empty
    tf.reset_default_graph()

    # Initialize the second training manager which should warm start from EMA
    # variables of `first/model.ckpt-100`.
    second = TrainingManager(join(top_dir, 'second.toml'))

    with tf.Graph().as_default():
        dataset = second.dataset
        nn = second.nn
        hparams = second.hparams

        if exists(hparams.train.model_dir):
            shutil.rmtree(hparams.train.model_dir)

        estimator = tf.estimator.Estimator(
            model_fn=nn.model_fn,
            warm_start_from=None,
            model_dir=hparams.train.model_dir,
            config=tf.estimator.RunConfig(
                save_checkpoints_steps=hparams.train.eval_steps,
                tf_random_seed=Defaults.seed,
                log_step_count_steps=None,
                keep_checkpoint_max=hparams.train.max_checkpoints_to_keep,
                session_config=tf.ConfigProto(allow_soft_placement=True)),
            params=hparams)

        train_spec = tf.estimator.TrainSpec(
            input_fn=dataset.input_fn(
                mode=tf.estimator.ModeKeys.TRAIN,
                batch_size=hparams.train.batch_size,
                shuffle=hparams.train.shuffle),
            max_steps=hparams.train.train_steps)

        hook = DebugWarmStartFromVariablesHook(
            previous_checkpoint=join(top_dir, 'first', 'model.ckpt-100'),
            true_values=values)

        estimator.train(
            input_fn=train_spec.input_fn,
            hooks=[hook, ],
            max_steps=1)


if __name__ == "__main__":
    nose.main()
