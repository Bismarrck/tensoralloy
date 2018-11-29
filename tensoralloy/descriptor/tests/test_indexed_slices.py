# coding=utf-8
"""
This module defines unit tests for `IndexTransformer`.
"""
from __future__ import print_function, absolute_import

import numpy as np
import nose
from collections import Counter
from unittest import TestCase
from nose.tools import assert_equal, assert_list_equal, assert_less

from tensoralloy.descriptor import IndexTransformer
from tensoralloy.misc import Pd3O2, Pd2O2Pd

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class IndexTransformerTest(TestCase):
    """
    A set of unit tests for the class `IndexTransformer`.
    """

    def setUp(self):
        """
        Setup the tests.
        """
        symbols = Pd3O2.get_chemical_symbols()
        self.max_occurs = Counter({'Pd': 4, 'O': 5})
        self.clf = IndexTransformer(self.max_occurs, symbols)

    def test_forward(self):
        assert_equal(len(self.clf.reference_symbols), 9)

        array = np.expand_dims([1, 2, 3, 4, 5], axis=1)
        results = self.clf.map_array(array, reverse=False).flatten().tolist()
        assert_list_equal(results, [0, 4, 5, 0, 0, 0, 1, 2, 3, 0])

        array = np.expand_dims([7, 1, 2, 3, 4, 5], axis=1)
        results = self.clf.map_array(array, reverse=False).flatten().tolist()
        assert_list_equal(results, [7, 4, 5, 7, 7, 7, 1, 2, 3, 7])

    def test_reverse(self):
        array = np.expand_dims([0, 4, 5, 0, 0, 0, 1, 2, 3, 0], axis=1)
        results = self.clf.map_array(array, reverse=True).flatten().tolist()
        assert_list_equal(results, [1, 2, 3, 4, 5])

    def test_call(self):
        assert_equal(self.clf.inplace_map_index(1), 6)
        assert_equal(self.clf.inplace_map_index(1, exclude_extra=True), 5)

    def test_mask(self):
        assert_list_equal(self.clf.mask.tolist(),
                          [0, 1, 1, 0, 0, 0, 1, 1, 1, 0])

    def test_permutation(self):
        symbols = Pd2O2Pd.get_chemical_symbols()
        clf = IndexTransformer(self.max_occurs, symbols)
        assert_list_equal(clf.mask.tolist(),
                          [0, 1, 1, 0, 0, 0, 1, 1, 1, 0])
        assert_less(np.abs(clf.map_array(Pd2O2Pd.positions) -
                           self.clf.map_array(Pd3O2.positions)).max(), 1e-8)


if __name__ == "__main__":
    nose.run()
