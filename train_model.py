# coding=utf-8
"""
This module is used to train the model
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from argparse import ArgumentParser
from config import ConfigParser
from utils import set_logging_configs
from os.path import join
from misc import check_path, Defaults

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def train_and_evaluate(config: ConfigParser):
    """
    Train and evaluate with the given configuration.

    This method is built upon `tr.estimator.train_and_evalutate`.

    """
    graph = tf.Graph()

    with graph.as_default():

        dataset = config.dataset
        nn = config.nn
        hparams = config.hyper_params

        set_logging_configs(
            logfile=check_path(join(hparams.train.model_dir, 'logfile')))

        estimator = tf.estimator.Estimator(
            model_fn=nn.model_fn,
            model_dir=hparams.train.model_dir,
            config=tf.estimator.RunConfig(
                save_checkpoints_steps=hparams.train.eval_steps,
                tf_random_seed=Defaults.seed,
                log_step_count_steps=hparams.train.log_steps,
                keep_checkpoint_max=hparams.train.max_checkpoints_to_keep,
                session_config=tf.ConfigProto(allow_soft_placement=True)),
            params=hparams)

        train_spec = tf.estimator.TrainSpec(
            input_fn=dataset.input_fn(
                mode=tf.estimator.ModeKeys.TRAIN,
                batch_size=hparams.train.batch_size,
                shuffle=True),
            max_steps=hparams.train.train_steps)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=dataset.input_fn(
                mode=tf.estimator.ModeKeys.EVAL,
                batch_size=hparams.train.batch_size,
                num_epochs=1,
                shuffle=False),
            steps=hparams.train.eval_steps,
            # Explicitly set these thresholds to lower values so that every
            # checkpoint can be evaluated.
            start_delay_secs=30,
            throttle_secs=60,
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        'filename',
        type=str,
        help="A cfg file to read."
    )
    train_and_evaluate(ConfigParser(parser.parse_args().filename))
