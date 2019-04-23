#!coding=utf-8
"""
This module defines unit tests of the data classes.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal, assert_raises

from tensoralloy.nn.dataclasses import OptParameters, LossParameters
from tensoralloy.utils import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_dataclasses():
    """
    Test some data classes.
    """
    hparams = AttributeDict(
        opt=AttributeDict(decay_steps=100, method='rmsprop'),
        loss=AttributeDict(
            energy=AttributeDict(per_atom_loss=True),
            elastic=AttributeDict(crystals=['Ni'],
                                  constraint=AttributeDict(stress_weight=1.5))
        )
    )

    opt_parameters = OptParameters(**hparams.opt)

    assert_equal(opt_parameters.decay_steps, 100)
    assert_equal(opt_parameters.method, 'rmsprop')

    loss_parameters = LossParameters(**hparams.loss)

    assert_equal(loss_parameters.energy.per_atom_loss, True)
    assert_equal(loss_parameters.elastic.crystals, ['Ni'])
    assert_equal(loss_parameters.elastic.constraint.stress_weight, 1.5)

    with assert_raises(Exception):
        _ = OptParameters(**AttributeDict(decay_method='adam'))


if __name__ == "__main__":
    nose.run()
