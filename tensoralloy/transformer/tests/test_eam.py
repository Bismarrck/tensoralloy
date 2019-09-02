# coding=utf-8
"""
This module defines unit tests of `BatchEAMTransformer`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from ase.atoms import Atoms
from ase.neighborlist import neighbor_list
from collections import Counter
from os.path import join

from tensoralloy.io.db import connect
from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.utils import get_kbody_terms, AttributeDict
from tensoralloy.test_utils import assert_array_equal, datasets_dir
from tensoralloy.transformer.vap import VirtualAtomMap
from tensoralloy.transformer.eam import BatchEAMTransformer, EAMTransformer
from tensoralloy.precision import precision_scope

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def eam(atoms: Atoms, max_occurs: Counter, nnl: int, rc=6.5):
    """
    A numpy based implementation of the Embedded-Atom Method.
    """
    symbols = atoms.get_chemical_symbols()
    ilist, jlist, dlist = neighbor_list('ijd', atoms, cutoff=rc)
    clf = VirtualAtomMap(max_occurs, symbols)

    kbody_terms = get_kbody_terms(sorted(max_occurs.keys()), angular=False)[1]
    max_n_atoms = sum(max_occurs.values()) + 1
    max_n_terms = max(map(len, kbody_terms.values()))

    counters = {atomi: Counter() for atomi in range(len(symbols))}

    g = np.zeros((1, max_n_terms, max_n_atoms, nnl))
    for k in range(len(ilist)):
        atomi = ilist[k]
        atomj = jlist[k]
        center = symbols[atomi]
        other = symbols[atomj]

        kbody_term = f'{center}{other}'

        idx1 = kbody_terms[center].index(kbody_term)
        idx2 = clf.local_to_gsl_map[atomi + 1]
        idx3 = counters[atomi][idx1]
        counters[atomi][idx1] += 1

        g[0, idx1, idx2, idx3] = dlist[k]

    mask = np.asarray(g > 0.0).astype(np.float64)
    return {'g': g, 'mask': mask}


def test_eam_transformer():
    """
    Test `EAMTransformer`.
    """
    db = connect(join(datasets_dir(), 'qm7.db'))
    atoms = db.get_atoms('id=2')
    rc = 6.5
    elements = ['C', 'H']
    max_occurs = Counter({'C': 2, 'H': 6})
    size = find_neighbor_size_of_atoms(atoms, rc=rc)
    ref = eam(atoms, max_occurs, nnl=size.nnl, rc=rc)

    with precision_scope('high'):
        with tf.Graph().as_default():

            clf = EAMTransformer(rc=6.5, elements=elements)

            with tf.Session() as sess:
                results = sess.run(
                    clf.get_descriptors(clf.get_constant_features(atoms)))

            g_h, mask_h = results['H']
            g_c, mask_c = results['C']

            assert_array_equal(ref['g'][0, :, 3: 9, :], g_h[0])
            assert_array_equal(ref['g'][0, :, 1: 3, :], g_c[0])
            assert_array_equal(ref['mask'][0, :, 3: 9, :], mask_h)
            assert_array_equal(ref['mask'][0, :, 1: 3, :], mask_c)


def test_batch_eam_transformer():
    """
    Test `BatchEAMTransformer`.
    """
    db = connect(join(datasets_dir(), 'qm7.db'))
    atoms = db.get_atoms('id=1')
    rc = 6.5
    max_occurs = Counter({'C': 2, 'H': 4})
    size = find_neighbor_size_of_atoms(atoms, rc=rc)
    nij_max = size.nij + 10
    nnl_max = size.nnl + 2
    ref = eam(atoms, max_occurs, nnl=nnl_max, rc=rc)

    with precision_scope('high'):
        with tf.Graph().as_default():

            clf = BatchEAMTransformer(
                rc=6.5, max_occurs=max_occurs, nij_max=nij_max, nnl_max=nnl_max)
            protobuf = tf.convert_to_tensor(
                clf.encode(atoms).SerializeToString())
            example = clf.decode_protobuf(protobuf)

            batch = AttributeDict()
            for key, tensor in example.items():
                batch[key] = tf.expand_dims(
                    tensor, axis=0, name=tensor.op.name + '/batch')

            ops = clf.get_descriptors(batch)

            with tf.Session() as sess:
                results = sess.run(ops)

            g_h, mask_h = results['H']
            g_c, mask_c = results['C']

            assert_array_equal(ref['g'][:, :, 3: 7, :], g_h[0])
            assert_array_equal(ref['g'][:, :, 1: 3, :], g_c[0])
            assert_array_equal(ref['mask'][:, :, 3: 7, :], mask_h)
            assert_array_equal(ref['mask'][:, :, 1: 3, :], mask_c)


def test_encode_atoms():
    """
    Test the method `BatchEAMTransformer.encode`.
    """
    db = connect(join(datasets_dir(), 'snap-Ni.db'))
    atoms = db.get_atoms('id=1')
    max_occurs = Counter({'Ni': len(atoms) + 1})
    rc = 6.0

    nij_max = db.get_nij_max(rc, allow_calculation=True)
    nnl_max = db.get_nnl_max(rc)

    with precision_scope('high'):
        with tf.Graph().as_default():
            clf = BatchEAMTransformer(rc=rc, max_occurs=max_occurs,
                                      nij_max=nij_max, nnl_max=nnl_max,
                                      use_stress=True)
            protobuf = tf.convert_to_tensor(clf.encode(atoms).SerializeToString())
            example = clf.decode_protobuf(protobuf)

            with tf.Session() as sess:
                results = sess.run(example)

            assert_array_equal(
                atoms.get_stress(voigt=True),
                results['stress'])


if __name__ == "__main__":
    nose.run()
