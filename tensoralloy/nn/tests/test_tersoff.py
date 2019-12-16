#!coding=utf-8
"""
Test the Tersoff potential.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose

from tensorflow_estimator import estimator as tf_estimator
from ase.build import bulk

from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.nn.tersoff import Tersoff
from tensoralloy.precision import precision_scope

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_tersoff_silicon():
    """
    Test the Tersoff potential with Si.
    """
    with tf.Graph().as_default():
        with precision_scope("high"):
            atoms = bulk('Si', cubic=True)
            elements = list(set(atoms.get_chemical_symbols()))
            nn = Tersoff(elements)
            clf = UniversalTransformer(
                elements, rcut=3.2, acut=3.2, angular=True, symmetric=False)
            nn.attach_transformer(clf)
            predictions = nn.build(
                features=clf.get_constant_features(atoms),
                mode=tf_estimator.ModeKeys.PREDICT)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                print(sess.run(predictions))


if __name__ == "__main__":
    nose.main()
