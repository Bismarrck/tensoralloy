# coding=utf-8
"""
This module defines unit tests of `AgSutton90`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose
import os

from tensorflow_estimator import estimator as tf_estimator
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from collections import Counter
from os.path import join, exists
from shutil import rmtree
from unittest import skipUnless
from nose.tools import assert_almost_equal
from nose import with_setup

from tensoralloy.transformer import BatchEAMTransformer, EAMTransformer
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.io.neighbor import find_neighbor_sizes
from tensoralloy.utils import AttributeDict
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# Setup the environment for `LAMMPS`
if 'LAMMPS_COMMAND' not in os.environ:
    LAMMPS_COMMAND = '/usr/local/bin/lmp_serial'
    os.environ['LAMMPS_COMMAND'] = LAMMPS_COMMAND
else:
    LAMMPS_COMMAND = os.environ['LAMMPS_COMMAND']


tmp_dir = join(test_dir(absolute=True), 'lammps', 'sutton90')


def get_lammps_calculator():
    """
    Return a LAMMPS calculator for Ag.
    """
    eam_file = join(test_dir(absolute=True), 'lammps', 'Ag.funcfl.eam')
    parameters = {'pair_style': 'eam',
                  'pair_coeff': ['1 1 Ag.funcfl.eam']}
    return LAMMPS(files=[eam_file], parameters=parameters, tmp_dir=tmp_dir,
                  keep_tmp_files=False, keep_alive=False, no_data_file=False)


def teardown():
    """
    Delete the tmp dir.
    """
    if exists(tmp_dir):
        rmtree(tmp_dir, ignore_errors=True)


@with_setup(teardown=teardown)
@skipUnless(exists(LAMMPS_COMMAND), f"{LAMMPS_COMMAND} not set!")
def test_eam_sutton90_batch_transformer():
    """
    Test the total energy calculation of `EamAlloyNN` with `AgSutton90` using
    `BatchEAMTransformer`
    """
    ref = bulk('Ag') * [2, 2, 2]
    rc = 11.999
    nij_max, _, nnl_max = find_neighbor_sizes(ref, rc=rc, k_max=2)
    max_occurs = Counter(ref.get_chemical_symbols())

    lammps = get_lammps_calculator()
    atoms = bulk('Ag') * [2, 2, 1]
    atoms.calc = lammps

    with tf.Graph().as_default():

        clf = BatchEAMTransformer(rc=rc, max_occurs=max_occurs, nij_max=nij_max,
                                  nnl_max=nnl_max)

        protobuf = tf.convert_to_tensor(clf.encode(atoms).SerializeToString())
        example = clf.decode_protobuf(protobuf)

        batch = AttributeDict()
        for key, tensor in example.items():
            batch[key] = tf.expand_dims(
                tensor, axis=0, name=tensor.op.name + '/batch')

        descriptors = clf.get_descriptors(batch)
        features = AttributeDict(positions=batch.positions,
                                 n_atoms=batch.n_atoms,
                                 cells=batch.cells,
                                 composition=batch.composition,
                                 mask=batch.mask,
                                 volume=batch.volume)

        nn = EamAlloyNN(elements=['Ag'], custom_potentials={
            "Ag": {"rho": "sutton90", "embed": "sutton90"},
            "AgAg": {"phi": "sutton90"}})
        outputs = nn._get_model_outputs(
            features=features,
            descriptors=AttributeDict(descriptors),
            mode=tf_estimator.ModeKeys.EVAL,
            verbose=False)
        prediction = AttributeDict(energy=nn._get_energy_op(outputs, features))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            energy = float(sess.run(prediction.energy))

        assert_almost_equal(energy,
                            lammps.get_potential_energy(atoms), delta=1e-6)


@with_setup(teardown=teardown)
@skipUnless(exists(LAMMPS_COMMAND), f"{LAMMPS_COMMAND} not set!")
def test_eam_sutton90():
    """
    Test the total energy calculation of `EamAlloyNN` with `AgSutton90` using
    `EAMTransformer`
    """
    rc = 11.999
    lammps = get_lammps_calculator()

    atoms = bulk('Ag') * [2, 2, 2]
    atoms.calc = lammps
    elements = ['Ag']

    with tf.Graph().as_default():
        clf = EAMTransformer(rc=rc, elements=elements)
        nn = EamAlloyNN(elements=elements,
                        custom_potentials={
                            "Ag": {"rho": "sutton90", "embed": "sutton90"},
                            "AgAg": {"phi": "sutton90"}})
        nn.attach_transformer(clf)
        prediction = nn.build(
            features=clf.get_placeholder_features(),
            mode=tf_estimator.ModeKeys.PREDICT,
            verbose=True)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            energy = float(sess.run(prediction.energy,
                                    feed_dict=clf.get_feed_dict(atoms)))

    atoms.calc = lammps
    assert_almost_equal(energy,
                        lammps.get_potential_energy(atoms), delta=1e-6)


if __name__ == "__main__":
    nose.run()
