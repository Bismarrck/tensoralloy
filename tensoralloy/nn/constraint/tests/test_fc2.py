#!coding=utf-8
"""
Unit tests for the force constants constraint loss.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from nose import main
from nose.tools import assert_less
from ase.build import bulk
from ase.io import write
from pathlib import Path

from tensoralloy.nn import EamAlloyNN
from tensoralloy.nn.constraint.data import get_crystal
from tensoralloy.nn.dataclasses import ForceConstantsLossOptions
from tensoralloy.utils import ModeKeys
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.nn.constraint.fc import reorder_phonopy_fc2
from tensoralloy.nn.constraint.fc import get_fc2_loss
from tensoralloy.precision import precision_scope
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_reorder_phonopy_fc2(save=False):
    with tf.Graph().as_default():
        with precision_scope("medium"):
            atoms = bulk("Ni", cubic=True) * [2, 2, 2]
            elements = ["Ni"]
            rc = 6.5
            nn = EamAlloyNN(elements, custom_potentials="zjw04",
                            export_properties=["energy", "forces", "hessian"])
            clf = UniversalTransformer(elements, rc)
            nn.attach_transformer(clf)
            vap = clf.get_vap_transformer(atoms)
            fc2_orig = nn.build(
                clf.get_constant_features(atoms), ModeKeys.PREDICT)["hessian"]
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                fc2_orig = sess.run(fc2_orig)

            fc2_phonopy = vap.reverse_map_hessian(
                fc2_orig, phonopy_format=True)
            fc2, weights = reorder_phonopy_fc2(fc2_phonopy, vap)
            assert_less(np.abs((fc2 - fc2_orig) * weights).max(), 1e-5)

            if save:
                np.save("Ni_fc2.npy", fc2_phonopy)
                write("Ni_sc.cif", atoms)


def test_fc2_loss():
    with tf.Graph().as_default():
        with precision_scope("medium"):
            root = Path(test_dir())
            crystal = get_crystal(root.joinpath("crystals").joinpath("Ni.toml"))
            elements = ['Ni']
            rc = 6.5
            nn = EamAlloyNN(elements, custom_potentials="zjw04")
            clf = UniversalTransformer(elements, rc)
            nn.attach_transformer(clf)
            options = ForceConstantsLossOptions(forces_weight=0.0,
                                                crystals=[crystal])
            loss = get_fc2_loss(nn, options)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                print(sess.run(loss))


if __name__ == "__main__":
    main()
