#!coding=utf-8
"""
Test the GRAP module.
"""
from __future__ import print_function, absolute_import

import nose
import tensorflow as tf
import numpy as np
from tensorflow_estimator import estimator
from ase.build import bulk
from nose.tools import assert_equal
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.nn.atomic.grap import Algorithm, GenericRadialAtomicPotential
from tensoralloy.test_utils import assert_array_almost_equal
from tensoralloy.precision import precision_scope

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_gen_algorithm():
    """
    Test the gen method of `Algorithm`.
    """
    params = {"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]}
    Algorithm.required_keys = ["x", "y"]
    algo = Algorithm(params, param_space_method="pair")
    assert_equal(len(algo), 5)

    for i in range(5):
        row = algo[i]
        assert_equal(row['x'], params['x'][i])
        assert_equal(row['y'], params['y'][i])

    Algorithm.required_keys = []


def test_serialization():
    grap = GenericRadialAtomicPotential(
        elements=['Be'], parameters={"eta": [1.0, 2.0], "omega": [0.0]})
    grap.as_dict()


def test_moment_tensor():
    with tf.Graph().as_default():
        with precision_scope("high"):
            rlist = [1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 1.9, 2.05, 2.2, 2.35]
            plist = [5.0, 4.75, 4.5, 4.25, 4.0, 3.75, 3.5, 3.25, 3.0, 2.75]
            elements = ['Be']
            grap1 = GenericRadialAtomicPotential(
                elements, "pexp", parameters={"rl": rlist, "pl": plist},
                param_space_method="pair",
                moment_tensors=[0, 1, 2],
                moment_scale_factors=1.0,
                cutoff_function="polynomial")
            grap2 = GenericRadialAtomicPotential(
                elements, "pexp", parameters={"rl": rlist, "pl": plist},
                param_space_method="pair",
                moment_tensors=[0, 1, 2],
                moment_scale_factors=[1.0, 100.0, 0.1],
                cutoff_function="polynomial")

            atoms = bulk('Be') * [2, 2, 2]
            atoms.positions += np.random.rand(16, 3) * 0.1
            clf = UniversalTransformer(elements, 5.0)

            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                op1 = grap1.calculate(
                    clf,
                    clf.get_descriptors(clf.get_placeholder_features()),
                    estimator.ModeKeys.PREDICT).descriptors
                op2 = grap2.calculate(
                    clf,
                    clf.get_descriptors(clf.get_placeholder_features()),
                    estimator.ModeKeys.PREDICT).descriptors
                g1, g2 = sess.run([op1, op2], feed_dict=clf.get_feed_dict(atoms))
                g1 = g1['Be'][0]
                g2 = g2['Be'][0]
                assert_array_almost_equal(
                    g1[:, 0:-1:3], g2[:, 0:-1:3], delta=1e-6)
                assert_array_almost_equal(
                    g1[:, 1:-1:3] * 100.0, g2[:, 1:-1:3], delta=1e-6)
                assert_array_almost_equal(
                    g1[:, 2:-1:3] * 0.1, g2[:, 2:-1:3], delta=1e-6)


if __name__ == "__main__":
    nose.main()
