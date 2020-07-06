# coding=utf-8
"""
This module defines tests of `AtomicNN` and its variants.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal, assert_list_equal

from tensoralloy.nn.atomic import AtomicNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_as_dict():
    """
    Test the method `AtomicNN.as_dict`.
    """
    elements = ['Al', 'Cu']
    hidden_sizes = 32
    old_nn = AtomicNN(elements, hidden_sizes,
                      activation='tanh',
                      use_atomic_static_energy=False,
                      minimize_properties=['energy', ],
                      export_properties=['energy', ])

    d = old_nn.as_dict()

    assert_equal(d['class'], 'AtomicNN')
    d.pop('class')

    new_nn = AtomicNN(**d)

    assert_list_equal(new_nn.elements, old_nn.elements)
    assert_list_equal(new_nn.minimize_properties, old_nn.minimize_properties)
    assert_equal(new_nn.hidden_sizes, old_nn.hidden_sizes)


if __name__ == "__main__":
    nose.main()
