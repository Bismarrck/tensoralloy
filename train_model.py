# coding=utf-8
"""
This module is used to train the model
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from argparse import ArgumentParser
from config import ConfigParser


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

        section = 'train'
        model_dir = config.get(section, 'model_dir')
        batch_size = config.getint(section, 'batch_size', fallback=50)
        train_steps = config.getint(section, 'train_steps', fallback=10000)
        eval_steps = config.getint(section, 'eval_steps', fallback=1000)

        estimator = tf.estimator.Estimator(
            model_fn=nn.model_fn, model_dir=model_dir, params=hparams)

        train_spec = tf.estimator.TrainSpec(
            input_fn=dataset.input_fn(batch_size=batch_size, shuffle=True),
            max_steps=train_steps)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=dataset.input_fn(batch_size=batch_size, shuffle=False),
            steps=eval_steps,
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
