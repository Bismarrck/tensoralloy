#!coding=utf-8
"""
Test the `UniversalTransformer`
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose

from ase.build import bulk

from tensoralloy.test_utils import assert_array_almost_equal
from tensoralloy.transformer.universal import UniversalTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_get_map():
    """
    Test the functions `get_g2_map` and `get_g4_map`.
    """
    atoms = bulk('Ni', cubic=True)

    with tf.Graph().as_default():
        clf = UniversalTransformer(['Ni'], rcut=4.5, acut=4.5, angular=True,
                                   use_computed_dists=True)
        descriptors = clf.build_graph(clf.get_np_features(atoms))
        with tf.Session() as sess:
            comput = sess.run(descriptors)

    with tf.Graph().as_default():
        clf = UniversalTransformer(['Ni'], rcut=4.5, acut=4.5, angular=True,
                                   use_computed_dists=False)
        descriptors = clf.build_graph(clf.get_np_features(atoms))
        with tf.Session() as sess:
            direct = sess.run(descriptors)

    eps = 1e-6
    assert_array_almost_equal(comput['radial']['Ni'][0],
                              direct['radial']['Ni'][0], delta=eps)
    assert_array_almost_equal(comput['radial']['Ni'][1],
                              direct['radial']['Ni'][1], delta=eps)
    assert_array_almost_equal(comput['angular']['Ni'][0],
                              direct['angular']['Ni'][0], delta=eps)
    assert_array_almost_equal(comput['angular']['Ni'][1],
                              direct['angular']['Ni'][1], delta=eps)


if __name__ == "__main__":
    nose.main()
