# coding=utf-8
"""
This module defines unit tests for the module `tensoralloy.nn.layers.layers`.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_dict_equal

from tensoralloy.nn.eam.potentials.potentials import EamEmpiricalPotential

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# noinspection PyAbstractClass
class _ParamPotential(EamEmpiricalPotential):
    """
    A layer with default parameters 'a', 'b', 'c' and 'd'.
    """

    defaults = {
        'AlCu': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0}
    }


def test_layer_init():
    """
    Test the initialization of `PotentialFunctionLayer`.
    """
    layer = EamEmpiricalPotential()

    assert_dict_equal(layer.params, {})
    assert_dict_equal(layer.fixed, {})

    layer = _ParamPotential()
    assert_dict_equal(layer.params, {'AlCu': {'a': 1.0, 'b': 1.0,
                                              'c': 1.0, 'd': 1.0}})

    layer = _ParamPotential(params={'AlCu': {"a": 2.0, "b": 3.0}})
    assert_dict_equal(layer.params,
                      {'AlCu': {'a': 2.0, 'b': 3.0, 'c': 1.0, 'd': 1.0}})


if __name__ == "__main__":
    nose.run()
