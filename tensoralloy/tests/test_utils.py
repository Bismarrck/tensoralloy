# coding=utf-8
"""
This module defines unit tests of `tensoralloy.utils`.
"""
from __future__ import print_function, absolute_import

import numpy as np
import nose

from nose.tools import assert_equal, assert_list_equal, assert_dict_equal

from tensoralloy.utils import szudzik_pairing_scalar, szudzik_pairing
from tensoralloy.utils import szudzik_pairing_nd
from tensoralloy.utils import szudzik_pairing_reverse
from tensoralloy.utils import cantor_pairing
from tensoralloy.utils import nested_get, nested_set
from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_szudzik_pairing():
    """
    Test the vectorized szudzik pairing function.
    """
    x = np.array([1, 3, 5, 8, 9])
    y = np.array([3, 4, 2, 2, 1])
    z = szudzik_pairing(x, y)
    for i, zi in enumerate(z):
        assert_equal(zi, szudzik_pairing_scalar(x[i], y[i]))

    w = szudzik_pairing_nd(x, y, z)
    for i, wi in enumerate(w):
        assert_equal(szudzik_pairing_nd(x[i], y[i], z[i]), wi)


def test_szudzik_pairing_reverse():
    """
    Test the reverse function of szudzik pairing.
    """
    x = np.array([1, 3, 5, 8, 9])
    y = np.array([3, 4, 2, 2, 1])
    z = szudzik_pairing(x, y)
    xx, yy = szudzik_pairing_reverse(z)
    for i in range(len(x)):
        assert_equal(xx[i], x[i])
        assert_equal(yy[i], y[i])


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


def test_cantor_pairing():
    """
    Test the pairing function: `cantor`.
    """
    x = np.arange(0, 100, 1)
    y = np.arange(0, 100, 1)
    xx, yy = np.meshgrid(x, y)
    z = cantor_pairing(xx.flatten(), yy.flatten()).tolist()
    assert_equal(len(set(z)), xx.size)


def test_get_elements_from_kbody_term():
    """
    Test `get_elements_from_kbody_term`.
    """
    assert_list_equal(get_elements_from_kbody_term("AlCu"), ['Al', 'Cu'])
    assert_list_equal(get_elements_from_kbody_term("CuH"), ['Cu', 'H'])
    assert_list_equal(get_elements_from_kbody_term("HH"), ['H', 'H'])
    assert_list_equal(get_elements_from_kbody_term("HHCu"), ['H', 'H', 'Cu'])
    assert_list_equal(get_elements_from_kbody_term("HHH"), ['H', 'H', 'H'])


def test_get_kbody_terms():
    """
    Test `get_kbody_terms`.
    """
    all_terms, kbody_terms_for_element, elements = get_kbody_terms(
        elements=['A'],
        angular=False)
    assert_list_equal(all_terms, ['AA'])
    assert_dict_equal(kbody_terms_for_element, {'A': ['AA']})

    all_terms, kbody_terms_for_element, elements = get_kbody_terms(
        elements=['A', 'B'],
        angular=False)
    assert_list_equal(all_terms, ['AA', 'AB', 'BB', 'BA'])
    assert_dict_equal(kbody_terms_for_element,
                      {'A': ['AA', 'AB'], 'B': ['BB', 'BA']})

    all_terms, kbody_terms_for_element, elements = get_kbody_terms(
        elements=['A', 'B'],
        angular=True)
    assert_list_equal(all_terms, ['AA', 'AB', 'AAA', 'AAB', 'ABB',
                                  'BB', 'BA', 'BAA', 'BAB', 'BBB'])
    assert_dict_equal(
        kbody_terms_for_element, {'A': ['AA', 'AB', 'AAA', 'AAB', 'ABB'],
                                  'B': ['BB', 'BA', 'BAA', 'BAB', 'BBB']})

    all_terms, kbody_terms_for_element, elements = get_kbody_terms(
        elements=['A', 'B', 'C'],
        angular=False)
    assert_list_equal(all_terms, ['AA', 'AB', 'AC',
                                  'BB', 'BA', 'BC',
                                  'CC', 'CA', 'CB'])
    assert_dict_equal(kbody_terms_for_element, {'A': ['AA', 'AB', 'AC'],
                                                'B': ['BB', 'BA', 'BC'],
                                                'C': ['CC', 'CA', 'CB']})

    all_terms, kbody_terms_for_element, elements = get_kbody_terms(
        elements=['A', 'B', 'C'],
        angular=True)
    assert_list_equal(all_terms, ['AA', 'AB', 'AC',
                                  'AAA', 'AAB', 'AAC', 'ABB', 'ABC', 'ACC',
                                  'BB', 'BA', 'BC',
                                  'BAA', 'BAB', 'BAC', 'BBB', 'BBC', 'BCC',
                                  'CC', 'CA', 'CB',
                                  'CAA', 'CAB', 'CAC', 'CBB', 'CBC', 'CCC'])
    assert_dict_equal(kbody_terms_for_element,
                      {'A': ['AA', 'AB', 'AC',
                             'AAA', 'AAB', 'AAC',
                             'ABB', 'ABC', 'ACC'],
                       'B': ['BB', 'BA', 'BC',
                             'BAA', 'BAB', 'BAC',
                             'BBB', 'BBC', 'BCC'],
                       'C': ['CC', 'CA', 'CB',
                             'CAA', 'CAB', 'CAC',
                             'CBB', 'CBC', 'CCC']})

    all_terms, kbody_terms_for_element, elements = get_kbody_terms(
        elements=['C', 'A', 'B'],
        angular=True,
        symmetric=False)
    assert_list_equal(all_terms, ['AA', 'AB', 'AC', 'AAA', 'AAB', 'AAC',
                                  'ABA', 'ABB', 'ABC', 'ACA', 'ACB', 'ACC',
                                  'BB', 'BA', 'BC', 'BAA', 'BAB', 'BAC',
                                  'BBA', 'BBB', 'BBC', 'BCA', 'BCB', 'BCC',
                                  'CC', 'CA', 'CB', 'CAA', 'CAB', 'CAC',
                                  'CBA', 'CBB', 'CBC', 'CCA', 'CCB', 'CCC'])
    assert_dict_equal(kbody_terms_for_element,
                      {'A': ['AA', 'AB', 'AC', 'AAA', 'AAB', 'AAC',
                             'ABA', 'ABB', 'ABC', 'ACA', 'ACB', 'ACC'],
                       'B': ['BB', 'BA', 'BC', 'BAA', 'BAB', 'BAC',
                             'BBA', 'BBB', 'BBC', 'BCA', 'BCB', 'BCC'],
                       'C': ['CC', 'CA', 'CB', 'CAA', 'CAB', 'CAC',
                             'CBA', 'CBB', 'CBC', 'CCA', 'CCB', 'CCC']})


if __name__ == "__main__":
    nose.run()
