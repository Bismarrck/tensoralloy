# coding=utf-8
"""
This module defines unit tests of `AgSutton90`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose
import os
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from collections import Counter
from os.path import join, exists
from shutil import rmtree
from unittest import skipUnless
from nose.tools import assert_almost_equal
from nose import with_setup

from tensoralloy.transformer import BatchEAMTransformer
from tensoralloy.nn.eam import EamNN
from tensoralloy.io.neighbor import find_neighbor_sizes
from tensoralloy.misc import AttributeDict, test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# Setup the environment for `LAMMPS`
LAMMPS_COMMAND = '/usr/local/bin/lmp_serial'
os.environ['LAMMPS_COMMAND'] = LAMMPS_COMMAND


def get_lammps_calc():
    """
    Return a LAMMPS calculator for Ag.
    """
    eam_file = join(test_dir(absolute=True), 'lammps', 'Ag.funcfl.eam')
    parameters = {'pair_style': 'eam',
                  'pair_coeff': ['1 1 Ag.funcfl.eam']}
    work_dir = join(test_dir(absolute=True), 'lammps', 'tmp')
    return LAMMPS(files=[eam_file], parameters=parameters, tmp_dir=work_dir,
                  keep_tmp_files=False, keep_alive=False, no_data_file=True)


LMP = get_lammps_calc()


def teardown():
    """
    Delete the tmp dir.
    """
    if exists(LMP.tmp_dir):
        rmtree(LMP.tmp_dir)


@with_setup(teardown=teardown)
@skipUnless(exists(LAMMPS_COMMAND), f"{LAMMPS_COMMAND} not exists!")
def test_eam_sutton90():
    """
    Test the total energy calculation of `EamNN` with `AgSutton90`.
    """
    ref = bulk('Ag') * [2, 2, 2]
    rc = 11.999
    nij_max, _, nnl_max = find_neighbor_sizes(ref, rc=rc, k_max=2)
    max_occurs = Counter(ref.get_chemical_symbols())

    atoms = bulk('Ag') * [2, 2, 1]
    atoms.calc = LMP

    with tf.Graph().as_default():

        clf = BatchEAMTransformer(rc=rc, max_occurs=max_occurs, nij_max=nij_max,
                                  nnl_max=nnl_max)

        protobuf = tf.convert_to_tensor(clf.encode(atoms).SerializeToString())
        example = clf.decode_protobuf(protobuf)

        batch = AttributeDict()
        for key, tensor in example.items():
            batch[key] = tf.expand_dims(
                tensor, axis=0, name=tensor.op.name + '/batch')

        ops = clf.get_descriptor_ops_from_batch(batch, batch_size=1)
        features = AttributeDict(descriptors=ops,
                                 positions=batch.positions,
                                 n_atoms=batch.n_atoms,
                                 cells=batch.cells,
                                 composition=batch.composition,
                                 mask=batch.mask,
                                 volume=batch.volume)

        nn = EamNN(elements=['Ag'], custom_layers={
            "Ag": "sutton90", "AgAg": {"rho": "sutton90", "phi": "sutton90"}})
        prediction = nn.build(features, verbose=False)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            energy = float(sess.run(prediction.energy))

        assert_almost_equal(energy, LMP.get_potential_energy(atoms), delta=1e-6)


if __name__ == "__main__":
    nose.run()
