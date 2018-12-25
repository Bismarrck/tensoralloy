# coding=utf-8
"""
This module defines unit tests of `tensoralloy.io.input.InputReader`.
"""
from __future__ import print_function, absolute_import

import nose
from nose.tools import assert_dict_equal, assert_equal, assert_list_equal
from nose.tools import assert_not_in
from os.path import join

from ..reader import nested_set, nested_get, InputReader
from tensoralloy.misc import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_nested_set():
    """
    Test the function `nested_set`.
    """
    d = {}
    nested_set(d, 'a.b.c', 4)
    assert_dict_equal(d, {'a': {'b': {'c': 4}}})

    nested_set(d, ['a', 'b'], {'x': [2, 3]})
    assert_dict_equal(d, {'a': {'b': {'x': [2, 3]}}})


def test_nested_get():
    """
    Test the function `nested_get`.
    """
    d = {
        'a': {
            'b': 2,
            'c': {
                'd': 4
            }
        }
    }
    assert_dict_equal(nested_get(d, "a.c"), {'d': 4})
    assert_equal(nested_get(d, 'a.b.c.d'), None)
    assert_equal(nested_get(d, ['a', 'c', 'd']), 4)


def test_read_configs():
    """
    Test `InputReader`.
    """
    reader = InputReader(join(test_dir(), 'inputs', 'qm7.behler.toml'))
    configs = reader.configs

    assert_equal(nested_get(configs, 'dataset.descriptor'), 'behler')
    assert_equal(nested_get(configs, 'nn.atomic.arch'), 'AtomicNN')
    assert_list_equal(nested_get(configs, 'nn.atomic.behler.eta'),
                      [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 20.0, 40.0])
    assert_equal(nested_get(configs, 'nn.atomic.behler.angular'), True)
    assert_not_in('eam', configs['nn'])
    assert_list_equal(nested_get(configs, 'nn.atomic.layers.C'), [64, 32])
    assert_list_equal(nested_get(configs, 'nn.atomic.layers.H'), [64, 32])
    assert_list_equal(nested_get(configs, 'nn.atomic.layers.N'), [64, 32])
    assert_list_equal(reader['nn.minimize'], ['energy'])
    assert_list_equal(reader['nn.atomic.export'], ['energy', 'forces'])

    reader = InputReader(join(test_dir(), 'inputs', 'qm7.alloy.eam.toml'))
    configs = reader.configs

    assert_not_in('behler', configs)
    assert_not_in('atomic', configs['nn'])
    assert_equal(len(nested_get(configs, 'nn.eam.rho')), 5)
    assert_equal(len(nested_get(configs, 'nn.eam.embed')), 5)
    assert_equal(reader['nn.loss.weight.l2'], 1.0)
    assert_equal(reader['nn.loss.weight.energy'], 0.0)

    reader = InputReader(join(test_dir(), 'inputs', 'AlFe.fs.eam.toml'))
    configs = reader.configs

    assert_equal(nested_get(configs, 'train.batch_size'), 50)
    assert_equal(nested_get(configs, 'train.shuffle'), True)
    assert_equal(nested_get(configs, 'nn.activation'), 'leaky_relu')
    assert_equal(nested_get(configs, 'nn.eam.export.lattice.type.Al'), 'fcc')
    assert_equal(nested_get(configs, 'nn.eam.export.lattice.type.Fe'), 'bcc')
    assert_equal(len(nested_get(configs, 'nn.eam.rho')), 4)
    assert_equal(len(nested_get(configs, 'nn.eam.embed')), 2)
    assert_list_equal(nested_get(configs, 'nn.eam.phi.AlFe'), [32, 32])
    assert_equal(reader['nn.eam.phi.AlAl'], 'msah11')
    assert_equal(reader['train.previous_checkpoint'], False)


if __name__ == "__main__":
    nose.run()
