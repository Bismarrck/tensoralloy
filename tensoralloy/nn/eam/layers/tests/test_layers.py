# coding=utf-8
"""
This module defines unit tests for the module `tensoralloy.nn.layers.layers`.
"""
from __future__ import print_function, absolute_import

import nose
from nose.tools import assert_list_equal, assert_dict_equal

from ..layers import PotentialFunctionLayer, any_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# noinspection PyAbstractClass
class _ParamLayer(PotentialFunctionLayer):
    """
    A layer with default parameters 'a', 'b', 'c' and 'd'.
    """

    defaults = {
        any_kbody_term: {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0}
    }


def test_layer_init():
    """
    Test the initialization of `PotentialFunctionLayer`.
    """
    layer = PotentialFunctionLayer()

    assert_list_equal(layer.allowed_kbody_terms, [any_kbody_term])
    assert_dict_equal(layer.params, {any_kbody_term: {}})
    assert_dict_equal(layer.fixed, {})

    layer = PotentialFunctionLayer(
        params={'AlCu': {"x": 2, "y": 3}}, fixed={'AlCu': ['x']})

    assert_dict_equal(layer.params, {any_kbody_term: {},
                                     'AlCu': {"x": 2, "y": 3}})
    assert_dict_equal(layer.fixed, {'AlCu': ['x']})

    layer = _ParamLayer()
    assert_dict_equal(layer.params, {any_kbody_term: {'a': 1.0, 'b': 1.0,
                                                      'c': 1.0, 'd': 1.0}})

    layer = _ParamLayer(params={'AlCu': {"a": 2.0, "b": 3.0}})
    assert_dict_equal(layer.params,
                      {'AlCu': {'a': 2.0, 'b': 3.0, 'c': 1.0, 'd': 1.0},
                       any_kbody_term: {'a': 1.0, 'b': 1.0,
                                        'c': 1.0, 'd': 1.0}})


if __name__ == "__main__":
    nose.run()
