#!coding=utf-8
"""
This module defines unit tests of the elastic constant module.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose
import os

from unittest import skipUnless
from ase.build import bulk
from nose.tools import assert_in, assert_almost_equal

from tensoralloy.transformer import UniversalTransformer
from tensoralloy.nn import EamAlloyNN
from tensoralloy.utils import ModeKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@skipUnless(os.environ.get('TEST_ELASTIC'),
            "The flag 'TEST_ELASTIC' is not set")
def test_elastic_constant_tensor_op():
    """
    Test the method `get_elastic_constant_tensor_op`.
    """
    graph = tf.Graph()
    with graph.as_default():
        elements = ['Ni']
        rc = 6.0
        export_properties = ['energy', 'forces', 'stress', 'elastic']
        nn = EamAlloyNN(elements, "zjw04", export_properties=export_properties)
        clf = UniversalTransformer(rcut=rc, elements=elements)
        nn.attach_transformer(clf)
        predictions = nn.build(clf.get_placeholder_features(),
                               mode=ModeKeys.PREDICT,
                               verbose=True)

        assert_in('elastic', predictions)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            atoms = bulk('Ni', cubic=True)
            elastic_tensor = sess.run(predictions["elastic"],
                                      feed_dict=clf.get_feed_dict(atoms))

        assert_almost_equal(elastic_tensor[0, 0], 247, delta=1)
        assert_almost_equal(elastic_tensor[1, 1], 247, delta=1)
        assert_almost_equal(elastic_tensor[2, 2], 247, delta=1)
        assert_almost_equal(elastic_tensor[0, 1], 147, delta=1)
        assert_almost_equal(elastic_tensor[0, 2], 147, delta=1)
        assert_almost_equal(elastic_tensor[3, 3], 125, delta=1)
        assert_almost_equal(elastic_tensor[4, 4], 125, delta=1)
        assert_almost_equal(elastic_tensor[5, 5], 125, delta=1)


if __name__ == "__main__":
    nose.run()
