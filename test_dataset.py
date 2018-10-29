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
from behler import SymmetryFunction
from misc import test_dir, Defaults, AttributeDict
from nose import main
from nose.tools import assert_equal, assert_list_equal, assert_dict_equal
from nose.tools import assert_less, assert_in, assert_true
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
    max_atoms = sum(max_occurs.values())
    sf = SymmetryFunction(Defaults.rc, max_occurs, k_max=2, nij_max=198,
                          nijk_max=0)
    rslices, _ = sf.get_indexed_slices(trajectory)
    positions = np.zeros((batch_size, max_atoms + 1, 3))
    clist = np.zeros((batch_size, 3, 3))
    for i, atoms in enumerate(trajectory):
        positions[i] = sf.get_index_transformer(atoms).gather(
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
    assert_true(dataset.load_tfrecords(savedir))

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
    tf.reset_default_graph()

    savedir = join(test_dir(), 'ethanol')
    database = connect(join(savedir, 'ethanol.db'))
    dataset = Dataset(database, 'ethanol', k_max=3)

    assert_equal(len(dataset), 10)
    assert_in(TrainableProperty.forces, dataset.trainable_properties)

    dataset.to_records(savedir, test_size=0.5)
    assert_true(dataset.load_tfrecords(savedir))

    # random_state: 611, test_size: 0.5 -> train: [0, 1, 8, 2, 3]
    next_batch = dataset.next_batch(mode=ModeKeys.TRAIN, batch_size=5,
                                    num_epochs=1, shuffle=False)

    atoms = database.get_atoms(id=2)
    clf = dataset.descriptor.get_index_transformer(atoms)
    positions = clf.gather(atoms.positions)
    energy = atoms.get_total_energy()
    forces = clf.gather(atoms.get_forces())

    with tf.Session() as sess:

        result = sess.run(next_batch)
        eps = 1e-8

        assert_less(np.abs(result.positions[1] - positions).max(), eps)
        assert_less(np.abs(result.f_true[1] - forces).max(), eps)
        assert_less(float(result.y_true[1] - energy), eps)


if __name__ == "__main__":
    main()
