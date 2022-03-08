#!coding=utf-8
"""
Test the GRAP module.
"""
from __future__ import print_function, absolute_import

import nose
import tensorflow as tf
import numpy as np
from ase.build import bulk
from nose.tools import assert_equal
from collections import Counter
from tensoralloy.utils import ModeKeys
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.neighbor import find_neighbor_size_of_atoms
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
        elements=['Be'], parameters={"eta": [1.0, 2.0], "omega": [0.0]}, 
        param_space_method="cross")
    grap.as_dict()


def test_grap_nn_algo():
    with tf.Graph().as_default():
        with precision_scope("high"):
            rlist = [1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 1.9, 2.05, 2.2, 2.35]
            plist = [5.0, 4.75, 4.5, 4.25, 4.0, 3.75, 3.5, 3.25, 3.0, 2.75]
            elements = ['Be', 'W']
            grap1 = GenericRadialAtomicPotential(
                elements, "pexp", parameters={"rl": rlist, "pl": plist},
                param_space_method="pair",
                legacy_mode=True,
                moment_tensors=[0, 1, 2],
                cutoff_function="polynomial")
            grap2 = GenericRadialAtomicPotential(
                elements, "pexp", parameters={"rl": rlist, "pl": plist},
                moment_tensors=[0, 1, 2],
                legacy_mode=False,
                cutoff_function="polynomial")
            atoms = bulk('Be') * [2, 2, 2]
            atoms.positions += np.random.rand(16, 3) * 0.1
            atoms.set_chemical_symbols(["Be"] * 8 + ["W"] * 8)

            rcut = 5.0
            neigh = find_neighbor_size_of_atoms(atoms, rc=rcut)
            max_occurs = Counter(atoms.get_chemical_symbols())

            clf = UniversalTransformer(elements, rcut=5.0)
            blf = BatchUniversalTransformer(
                max_occurs, rcut=rcut, nij_max=neigh.nij, nnl_max=neigh.nnl,
                batch_size=1)
            protobuf = tf.convert_to_tensor(
                blf.encode(atoms).SerializeToString())
            example = blf.decode_protobuf(protobuf)
            batch = dict()
            for key, tensor in example.items():
                batch[key] = tf.expand_dims(
                    tensor, axis=0, name=tensor.op.name + '/batch')

            descriptors = blf.get_descriptors(batch)

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                op1 = grap1.calculate(
                    clf,
                    clf.get_descriptors(clf.get_placeholder_features()),
                    ModeKeys.PREDICT).descriptors
                op2 = grap2.calculate(
                    clf,
                    clf.get_descriptors(clf.get_placeholder_features()),
                    ModeKeys.PREDICT).descriptors
                op3 = grap2.calculate(
                    blf, descriptors, ModeKeys.TRAIN).descriptors
                g1, g2, g3 = sess.run([op1, op2, op3],
                                      feed_dict=clf.get_feed_dict(atoms))
                assert_array_almost_equal(g1['Be'][0], g2['Be'][0], delta=1e-6)
                assert_array_almost_equal(g1['W'][0], g2['W'][0], delta=1e-6)
                assert_array_almost_equal(g1['Be'][0], g3['Be'][0], delta=1e-6)
                assert_array_almost_equal(g1['W'][0], g3['W'][0], delta=1e-6)



if __name__ == "__main__":
    nose.main()
