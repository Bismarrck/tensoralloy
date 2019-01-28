# coding=utf-8
"""
This module defines unit tests for `IndexTransformer`.
"""
from __future__ import print_function, absolute_import

import numpy as np
import nose

from collections import Counter
from unittest import TestCase
from os.path import join
from ase.build import bulk
from nose.tools import assert_equal, assert_list_equal, assert_less

from tensoralloy.transformer.index_transformer import IndexTransformer
from tensoralloy.test_utils import Pd3O2, Pd2O2Pd, test_dir
from tensoralloy.calculator import TensorAlloyCalculator

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
        assert_equal(len(self.clf.reference_chemical_symbols), 9)
        assert_equal(len(self.clf.chemical_symbols), 5)
        assert_equal(self.clf.n_atoms, 5)
        assert_equal(self.clf.max_n_atoms, 9)

        array = np.expand_dims([1, 2, 3, 4, 5], axis=1)
        results = self.clf.map_array(array, reverse=False).flatten().tolist()
        assert_list_equal(results, [0, 4, 5, 0, 0, 0, 1, 2, 3, 0])

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


def test_reverse_map_hessian():
    """
    Test the method `IndexTransformer.reverse_map_hessian()`.
    """
    graph_model_path = join(test_dir(), 'Ni', 'Ni.zjw04xc.pb')
    calc = TensorAlloyCalculator(graph_model_path)
    atoms = bulk('Ni', crystalstructure='fcc', cubic=True) * [2, 2, 2]
    atoms.calc = calc
    calc.calculate(atoms)
    original = calc.get_property('hessian', atoms)

    clf = calc.transformer.get_index_transformer(atoms)

    h = clf.reverse_map_hessian(original, phonopy_format=False)
    assert_list_equal(list(h.shape), [96, 96])

    h = clf.reverse_map_hessian(original, phonopy_format=True)
    assert_list_equal(list(h.shape), [32, 32, 3, 3])


if __name__ == "__main__":
    nose.run()
