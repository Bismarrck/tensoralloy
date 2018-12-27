# coding=utf-8
"""
This module defines unit tests of `BasicNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from nose.tools import assert_dict_equal, assert_equal
from ase.db import connect
from os.path import join

from tensoralloy.nn.basic import BasicNN
from tensoralloy.misc import Defaults, datasets_dir, AttributeDict
from tensoralloy.test_utils import assert_array_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_hidden_sizes():
    """
    Test setting hidden sizes of `BasicNN`.
    """
    elements = sorted(['Al', 'Cu'])

    nn = BasicNN(elements)
    assert_dict_equal(nn.hidden_sizes, {'Al': Defaults.hidden_sizes,
                                        'Cu': Defaults.hidden_sizes})

    nn = BasicNN(elements, hidden_sizes=32)
    assert_dict_equal(nn.hidden_sizes, {'Al': [32],
                                        'Cu': [32]})

    nn = BasicNN(elements, hidden_sizes=[64, 32])
    assert_dict_equal(nn.hidden_sizes, {'Al': [64, 32],
                                        'Cu': [64, 32]})

    nn = BasicNN(elements, hidden_sizes={'Al': [32, 16]})
    assert_dict_equal(nn.hidden_sizes, {'Al': [32, 16],
                                        'Cu': Defaults.hidden_sizes})

    nn = BasicNN(elements, hidden_sizes={'Al': [32, 32]})
    assert_dict_equal(nn.hidden_sizes, {'Al': [32, 32],
                                        'Cu': Defaults.hidden_sizes})


def test_convert_to_voigt_stress():
    """
    Test the method `BasicNN._convert_to_voigt_stress`.
    """
    db = connect(join(datasets_dir(), 'Ni.db'))
    nn = BasicNN(elements=['Ni'], activation='leaky_relu',
                 minimize_properties=['stress'], export_properties=['stress'])

    with tf.Graph().as_default():

        batch_size = 4
        batch = np.zeros((batch_size, 3, 3), dtype=np.float64)
        batch_voigt = np.zeros((batch_size, 6), dtype=np.float64)

        for i, index in enumerate(range(1, 5)):
            atoms = db.get_atoms(f'id={index}')
            batch[i] = atoms.get_stress(voigt=False)
            batch_voigt[i] = atoms.get_stress(voigt=True)

        op1 = nn._convert_to_voigt_stress(
            tf.convert_to_tensor(batch), tf.convert_to_tensor(batch_size))
        op2 = nn._convert_to_voigt_stress(
            tf.convert_to_tensor(batch[0]), batch_size=None)

        with tf.Session() as sess:
            pred_batch_voigt, pred_voigt = sess.run([op1, op2])

        assert_array_equal(pred_voigt, batch_voigt[0])
        assert_array_equal(pred_batch_voigt, batch_voigt)


def test_check_hparams():
    """
    Test the static method `BasicNN._check_hparams`.
    """
    nn = BasicNN(elements=['Ni'], activation='leaky_relu',
                 minimize_properties=['stress'], export_properties=['stress'])
    
    defaults = AttributeDict(
        loss=AttributeDict(
            energy=AttributeDict(weight=1.0, per_atom_loss=False),
            forces=AttributeDict(weight=1.0),
            stress=AttributeDict(weight=1.0),
            total_pressure=AttributeDict(weight=1.0),
            l2=AttributeDict(weight=0.01)))

    # noinspection PyTypeChecker
    hparams = nn._check_loss_hparams(None)
    assert_dict_equal(defaults, hparams)
    
    hparams = nn._check_loss_hparams(AttributeDict())
    assert_dict_equal(defaults, hparams)

    hparams = nn._check_loss_hparams(AttributeDict(loss={}))
    assert_dict_equal(defaults, hparams)

    hparams = nn._check_loss_hparams(
        AttributeDict(loss={'energy': {'weight': 3.0}}))
    assert_equal(hparams.loss.energy.weight, 3.0)
    assert_equal(hparams.loss.forces.weight, 1.0)


if __name__ == "__main__":
    nose.run()
