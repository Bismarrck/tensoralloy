#!coding=utf-8
"""
This module defines unit tests of the elastic constant module.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose

from tensorflow_estimator import estimator as tf_estimator
from ase.build import bulk
from nose.tools import assert_in, assert_almost_equal, assert_equal
from os.path import join

from tensoralloy.transformer import EAMTransformer
from tensoralloy.nn import EamAlloyNN
from tensoralloy.nn.elastic import read_external_crystal, built_in_crystals
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_read_external_crystal():
    """
    Test the function `read_external_crystal`.
    """
    toml_file = join(test_dir(), "crystals", "Ni3Mo_elastic_tensor.toml")
    crystal = read_external_crystal(toml_file)
    builtin = built_in_crystals['Ni3Mo']

    assert_equal(builtin.name, crystal.name)
    assert_equal(builtin.tag, crystal.tag)
    assert_equal(7, len(crystal.elastic_constants))

    for elastic_constant in crystal.elastic_constants:
        assert_in(elastic_constant, builtin.elastic_constants)


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
        clf = EAMTransformer(rc, elements)
        nn.attach_transformer(clf)
        predictions = nn.build(clf.get_placeholder_features(),
                               mode=tf_estimator.ModeKeys.PREDICT,
                               verbose=True)

        assert_in('elastic', predictions)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            atoms = bulk('Ni', cubic=True)
            elastic_tensor = sess.run(predictions.elastic,
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
