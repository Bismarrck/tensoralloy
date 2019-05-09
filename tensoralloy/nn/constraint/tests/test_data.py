#!coding=utf-8
"""
The unit tests of the built-in constraint data module.
"""
from __future__ import print_function, absolute_import

import nose

from os.path import join
from nose.tools import assert_equal, assert_in

from tensoralloy.nn.constraint.data import built_in_crystals
from tensoralloy.nn.constraint.elastic import read_external_crystal
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_read_external_crystal():
    """
    Test the function `read_external_crystal`.
    """
    toml_file = join(test_dir(), "crystals", "Ni3Mo_elastic_tensor.toml")
    crystal = read_external_crystal(toml_file)
    builtin = built_in_crystals['Ni3Mo']

    assert_equal(builtin.name, crystal.name)
    assert_equal(builtin.phase, crystal.phase)
    assert_equal(7, len(crystal.elastic_constants))

    for elastic_constant in crystal.elastic_constants:
        assert_in(elastic_constant, builtin.elastic_constants)

    assert_equal(crystal.elastic_constants[-1].weight, 0.0)


if __name__ == "__main__":
    nose.run()
