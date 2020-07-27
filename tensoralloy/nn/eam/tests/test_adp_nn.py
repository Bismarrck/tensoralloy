#!coding=utf-8
"""
This module defines unit tests of `AdpNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from tensorflow_estimator import estimator as tf_estimator
from os.path import join
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from nose.tools import assert_equal
from collections import Counter

from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.test_utils import data_dir
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.transformer import BatchUniversalTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_dynamic_partition():
    """
    Test the modified implementation of `_dynamic_partition`.
    """
    atoms = read(join(data_dir(), "crystals",
                      "Ni4Mo_mp-11507_conventional_standard.cif"))
    atoms.calc = SinglePointCalculator(
        atoms, **{'energy': 0.0, 'forces': np.zeros_like(atoms.positions)})

    rc = 6.5
    elements = ['Mo', 'Ni']

    with tf.Graph().as_default():

        clf = UniversalTransformer(rcut=rc, elements=elements)

        with tf.name_scope("Symmetric"):

            op, max_occurs = dynamic_partition(
                clf.get_descriptors(clf.get_placeholder_features())["radial"],
                elements=clf.elements,
                kbody_terms_for_element=clf.kbody_terms_for_element,
                mode=tf_estimator.ModeKeys.PREDICT,
                merge_symmetric=True)

            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                partitions = sess.run(
                    op, feed_dict=clf.get_feed_dict(atoms))

                assert_equal(len(partitions), 3)
                for key, (descriptor, mask) in partitions.items():
                    assert_equal(descriptor.shape[0], 4)
                    assert_equal(mask.shape[0], 1)

    with tf.Graph().as_default():
        size = find_neighbor_size_of_atoms(atoms, rc)
        max_occurs = Counter(atoms.get_chemical_symbols())

        clf = BatchUniversalTransformer(
            rcut=rc, max_occurs=max_occurs, nij_max=size.nij, nnl_max=size.nnl,
            batch_size=1)

        protobuf = tf.convert_to_tensor(clf.encode(atoms).SerializeToString())
        example = clf.decode_protobuf(protobuf)

        batch = dict()
        for key, tensor in example.items():
            batch[key] = tf.expand_dims(
                tensor, axis=0, name=tensor.op.name + '/batch')

        descriptors = clf.get_descriptors(batch)
        op, max_occurs = dynamic_partition(
            descriptors["radial"],
            elements=clf.elements,
            kbody_terms_for_element=clf.kbody_terms_for_element,
            mode=tf_estimator.ModeKeys.TRAIN,
            merge_symmetric=False)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            partitions = sess.run(op)

            assert_equal(len(partitions), 4)
            for key, (descriptor, mask) in partitions.items():
                assert_equal(descriptor.shape[0], 4)
                assert_equal(mask.shape[0], 1)


if __name__ == "__main__":
    nose.run()
