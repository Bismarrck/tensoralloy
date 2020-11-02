#!coding=utf-8
"""
Test the GRAP module.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal

from tensoralloy.nn.atomic.grap import Algorithm, GenericRadialAtomicPotential

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_gen_algorithm():
    """
    Test the gen method of `Algorithm`.
    """
    params = {"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]}
    Algorithm.required_keys = ["x", "y"]
    algo = Algorithm(params, param_space_method="pair")
    assert_equal(len(algo), 5)

    for i in range(5):
        row = algo[i]
        assert_equal(row['x'], params['x'][i])
        assert_equal(row['y'], params['y'][i])

    Algorithm.required_keys = []


def test_serialization():
    grap = GenericRadialAtomicPotential(
        elements=['Be'], parameters={"eta": [1.0, 2.0], "omega": [0.0]})
    grap.as_dict()


if __name__ == "__main__":
    nose.main()
