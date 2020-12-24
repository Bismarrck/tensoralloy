#!coding=utf-8
"""
Unit tests of `tensoralloy.neighbor` module.
"""
from __future__ import print_function, absolute_import

import nose

from os.path import join
from ase.db import connect
from nose.tools import assert_equal, assert_greater_equal

from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_find_sizes():
    """
    Test the function `_find_sizes`.
    """
    db = connect(join(test_dir(), 'datasets', 'qm7m', 'qm7m.db'))

    atoms = db.get_atoms('id=2')
    size = find_neighbor_size_of_atoms(atoms, 6.5, find_nijk=False)
    assert_equal(size.nij, 20)
    assert_equal(size.nijk, 0)
    assert_equal(size.nnl, 4)

    atoms = db.get_atoms('id=3')
    size = find_neighbor_size_of_atoms(atoms, 6.5, find_nijk=True)
    assert_equal(size.nij, 56)
    assert_equal(size.nijk, 168)
    assert_equal(size.nnl, 6)


def test_find_ij2k():
    """
    Test the calculation of `ij2k`.
    """
    from collections import Counter

    from tensoralloy.io.db import snap
    from tensoralloy.utils import get_kbody_terms, get_elements_from_kbody_term
    from tensoralloy.transformer.vap import VirtualAtomMap
    from tensoralloy.transformer.universal import get_radial_metadata
    from tensoralloy.transformer.universal import get_angular_metadata
    from tensoralloy.utils import ModeKeys

    db = snap()
    symmetric = False
    rc = 4.5

    for atoms_id in (1, 2, 3, 100, 200, 500, 1000, 1200, 2000, 3000):

        atoms = db.get_atoms(id=atoms_id)
        symbols = atoms.get_chemical_symbols()
        max_occurs = Counter(symbols)
        all_kbody_terms, kbody_terms_for_element, elements = \
            get_kbody_terms(["Mo", "Ni"], angular=True, symmetric=symmetric)
        radial_interactions = {}
        angular_interactions = {}
        for element in elements:
            kbody_terms = kbody_terms_for_element[element]
            for i, kbody_term in enumerate(kbody_terms):
                if len(get_elements_from_kbody_term(kbody_term)) == 2:
                    radial_interactions[kbody_term] = i
                else:
                    angular_interactions[kbody_term] = i - len(elements)
        vap = VirtualAtomMap(max_occurs, symbols)
        g2, ijn_id_map = get_radial_metadata(atoms,
                                             rc=rc,
                                             interactions=radial_interactions,
                                             vap=vap,
                                             mode=ModeKeys.PREDICT)
        g4 = get_angular_metadata(atoms,
                                  rc=rc,
                                  radial_interactions=radial_interactions,
                                  angular_interactions=angular_interactions,
                                  vap=vap,
                                  mode=ModeKeys.PREDICT,
                                  g2=g2,
                                  ijn_id_map=ijn_id_map)

        ij2k = g4.v2g_map[:, 3].max() + 1
        nl = find_neighbor_size_of_atoms(atoms, rc, find_ij2k=True)
        # TODO: a special case 'atoms_id=3000, nl.ij2k > ij2k' should be fixed
        assert_greater_equal(nl.ij2k, ij2k)


if __name__ == "__main__":
    nose.run()
