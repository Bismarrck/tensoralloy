# coding=utf-8
"""
This module defines unit tests of `Dataset`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import ModeKeys
from ase.db import connect
from ase.io import read
from dataset import Dataset, TrainableProperty
from behler import compute_dimension, get_kbody_terms
from behler import NeighborIndexBuilder
from misc import test_dir, Defaults, AttributeDict
from nose import main
from nose.tools import assert_equal, assert_list_equal, assert_dict_equal
from nose.tools import assert_less
from os.path import join
from collections import Counter

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def qm7m_compute():
    """
    Compute the reference values.
    """
    trajectory = read('test_files/qm7m/qm7m.xyz', index=':', format='xyz')
    max_occurs = Counter()
    for atoms in trajectory:
        for element, n in Counter(atoms.get_chemical_symbols()).items():
            max_occurs[element] = max(max_occurs[element], n)

    batch_size = len(trajectory)
    n_atoms = sum(max_occurs.values())
    kbody_terms, mapping, elements = get_kbody_terms(
        list(max_occurs.keys()), k_max=2
    )
    total_dim, kbody_sizes = compute_dimension(kbody_terms,
                                               Defaults.n_etas,
                                               Defaults.n_betas,
                                               Defaults.n_gammas,
                                               Defaults.n_zetas)

    nl = NeighborIndexBuilder(Defaults.rc, kbody_terms, kbody_sizes,
                              max_occurs, Defaults.n_etas, k_max=2,
                              nij_max=198, nijk_max=0)

    rslices, _ = nl.get_indexed_slices(trajectory)
    positions = np.zeros((batch_size, n_atoms + 1, 3))
    clist = np.zeros((batch_size, 3, 3))
    for i, atoms in enumerate(trajectory):
        positions[i] = nl.get_index_transformer(atoms).gather(
            atoms.positions)
        clist[i] = atoms.cell

    return AttributeDict(positions=positions, clist=clist, rslices=rslices)


def test_qm7m():
    """
    qm7m is a minimal subset of the `qm7` dataset. This dataset only has three
    molecules. Forces are not provided.
    """
    ref = qm7m_compute()

    tf.reset_default_graph()

    savedir = join(test_dir(), 'qm7m')
    database = connect(join(savedir, 'qm7m.db'))
    dataset = Dataset(database, 'qm7m', k_max=2)

    assert_equal(len(dataset), 3)
    assert_list_equal(dataset.trainable_properties, [TrainableProperty.energy])
    assert_dict_equal(dataset.max_occurs, {'C': 5, 'H': 8, 'O': 2})

    dataset.to_records(savedir, test_size=0.33)
    dataset.load_tfrecords(savedir)

    # random_state: 611, test_size: 0.33 -> train: 1, 2, test: 0
    next_batch = dataset.next_batch(mode=ModeKeys.EVAL, batch_size=1,
                                    num_epochs=1, shuffle=False)

    with tf.Session() as sess:

        result = sess.run(next_batch)
        eps = 1e-8

        assert_equal(len(result.keys()), 7)
        assert_less(np.abs(result.positions[0] - ref.positions[0]).max(), eps)
        assert_less(np.abs(result.cell[0] - ref.clist[0]).max(), eps)
        assert_less(np.abs(result.rv2g[0] - ref.rslices.v2g_map[0]).max(), eps)
        assert_less(np.abs(result.ilist[0] - ref.rslices.ilist[0]).max(), eps)
        assert_less(np.abs(result.Slist[0] - ref.rslices.Slist[0]).max(), eps)


def test_ethanol():
    """
    This is a minimal subset of the ethanol MD dataset with 10 configurations.
    Forces are provided.
    """
    pass


if __name__ == "__main__":
    main()
