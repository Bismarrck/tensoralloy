#!coding=utf-8
"""
The unit tests of DeePMD.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

import tensorflow as tf
import nose

from nose.tools import assert_list_equal
from collections import Counter
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.transformer import DeePMDTransformer, BatchDeePMDTransformer
from tensoralloy.nn.atomic.deepmd import DeepPotSE
from tensoralloy.io.db import snap
from tensoralloy.neighbor import find_neighbor_size_of_atoms


def test_init():
    """
    Test the initialization of a DeepPotSE for prediction.
    """
    with tf.Graph().as_default():
        rc = 6.0
        rcs = 4.0
        elements = ["Mo", "Ni"]
        atoms = snap().get_atoms(id=1)

        clf = DeePMDTransformer(rc, elements)
        nn = DeepPotSE(elements, rcs, hidden_sizes=[64, 32],
                       activation='softplus')
        nn.attach_transformer(clf)
        predictions = nn.build(features=clf.get_constant_features(atoms),
                               mode=tf_estimator.ModeKeys.PREDICT,
                               verbose=True)

        assert_list_equal(predictions["energy"].shape.as_list(), [])
        assert_list_equal(predictions["forces"].shape.as_list(),
                          [len(atoms), 3])


def test_batch_init():
    """
    Test the initialization of a DeepPotSE for training.
    """
    with tf.Graph().as_default():
        rc = 6.0
        rcs = 4.0
        elements = ["Mo", "Ni"]
        atoms = snap().get_atoms(id=1)

        nn = DeepPotSE(elements, rcs, activation='softplus',
                       hidden_sizes=[64, 32])

        size = find_neighbor_size_of_atoms(atoms, rc)
        max_occurs = Counter(atoms.get_chemical_symbols())

        clf = BatchDeePMDTransformer(rc, max_occurs, size.nij, size.nnl, 1)

        protobuf = tf.convert_to_tensor(clf.encode(atoms).SerializeToString())
        example = clf.decode_protobuf(protobuf)
        batch = dict()
        for key, tensor in example.items():
            batch[key] = tf.expand_dims(
                tensor, axis=0, name=tensor.op.name + '/batch')

        nn.attach_transformer(clf)
        predictions = nn.build(batch, verbose=True)
        print(predictions)


if __name__ == "__main__":
    nose.main()
