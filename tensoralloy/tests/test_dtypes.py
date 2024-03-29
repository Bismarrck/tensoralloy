#!coding=utf-8
"""
The unit tests of `tensoralloy.dtypes` module.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal

from tensoralloy.precision import precision_scope, get_float_precision, Precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_set_precision():
    """
    Test the with statement of `set_precision`.
    """
    with precision_scope(Precision.medium):
        assert_equal(get_float_precision(), Precision.medium)

    with precision_scope(Precision.high):
        assert_equal(get_float_precision(), Precision.high)


if __name__ == "__main__":
    nose.run()
