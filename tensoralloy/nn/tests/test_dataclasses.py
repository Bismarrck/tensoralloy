#!coding=utf-8
"""
This module defines unit tests of the data classes.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal, assert_raises, assert_is_none

from tensoralloy.nn.dataclasses import OptParameters, LossParameters
from tensoralloy.nn.dataclasses import TrainParameters
from tensoralloy.utils import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_dataclasses():
    """
    Test some data classes.
    """
    hparams = AttributeDict(
        opt=dict(decay_steps=100, method='rmsprop'),
        loss=dict(
            energy=dict(per_atom_loss=True),
            forces=dict(method='logcosh'),
            elastic=dict(crystals=['Ni'], constraint=dict(stress_weight=1.5))
        ),
        train=dict(model_dir="..",
                   ckpt=dict(restore_all_variables=False,
                             use_ema_variables=False))
    )

    opt_parameters = OptParameters(**hparams.opt)

    assert_equal(opt_parameters.decay_steps, 100)
    assert_equal(opt_parameters.method, 'rmsprop')

    loss_parameters = LossParameters(**hparams.loss)

    assert_equal(loss_parameters.energy.per_atom_loss, True)
    assert_equal(loss_parameters.forces.method, 'logcosh')
    assert_equal(loss_parameters.elastic.crystals, ['Ni'])
    assert_equal(loss_parameters.elastic.constraint.stress_weight, 1.5)

    assert_is_none(loss_parameters.rose.crystals, None)
    assert_equal(loss_parameters.rose.dx, 0.10)

    train_parameters = TrainParameters(**hparams['train'])
    assert_equal(train_parameters.ckpt.use_previous_ema_variables, False)

    with assert_raises(Exception):
        _ = OptParameters(**AttributeDict(decay_method='adam'))


if __name__ == "__main__":
    nose.run()
