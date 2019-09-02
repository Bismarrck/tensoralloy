# coding=utf-8
"""
This module defines unit tests of `AgSutton90`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose
import os
import unittest

from tensorflow_estimator import estimator as tf_estimator
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from collections import Counter
from os.path import join, exists
from shutil import rmtree
from unittest import skipUnless
from nose.tools import assert_almost_equal

from tensoralloy.transformer import BatchEAMTransformer, EAMTransformer
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.test_utils import test_dir
from tensoralloy.io.lammps import LAMMPS_COMMAND

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@skipUnless(exists(LAMMPS_COMMAND), f"LAMMPS not found!")
class EamSutton90Test(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        self.tmp_dir = join(test_dir(absolute=True), 'lammps', 'sutton90')
        if not exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def get_lammps_calculator(self):
        """
        Return a LAMMPS calculator for Ag.
        """
        eam_file = join(test_dir(absolute=True), 'lammps', 'Ag.funcfl.eam')
        return LAMMPS(files=[eam_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=self.tmp_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="eam",
                      command=LAMMPS_COMMAND,
                      pair_coeff=['1 1 Ag.funcfl.eam'])

    def tearDown(self):
        """
        Delete the tmp dir.
        """
        if exists(self.tmp_dir):
            rmtree(self.tmp_dir, ignore_errors=True)

    def test_eam_sutton90_batch_transformer(self):
        """
        Test the total energy calculation of `EamAlloyNN` with `AgSutton90`
        using `BatchEAMTransformer`.
        """
        ref = bulk('Ag') * [2, 2, 2]
        rc = 11.999
        size = find_neighbor_size_of_atoms(ref, rc=rc)
        max_occurs = Counter(ref.get_chemical_symbols())

        lammps = self.get_lammps_calculator()
        atoms = bulk('Ag') * [2, 2, 1]
        atoms.calc = lammps

        with tf.Graph().as_default():

            clf = BatchEAMTransformer(rc=rc,
                                      max_occurs=max_occurs,
                                      nij_max=size.nij,
                                      nnl_max=size.nnl)

            protobuf = tf.convert_to_tensor(
                clf.encode(atoms).SerializeToString())
            example = clf.decode_protobuf(protobuf)

            batch = dict()
            for key, tensor in example.items():
                batch[key] = tf.expand_dims(
                    tensor, axis=0, name=tensor.op.name + '/batch')

            descriptors = clf.get_descriptors(batch)
            features = dict(positions=batch["positions"],
                            n_atoms=batch["n_atoms"],
                            cell=batch["cell"],
                            compositions=batch["compositions"],
                            atom_masks=batch["atom_masks"],
                            volume=batch["volume"])

            nn = EamAlloyNN(elements=['Ag'], custom_potentials={
                "Ag": {"rho": "sutton90", "embed": "sutton90"},
                "AgAg": {"phi": "sutton90"}})
            outputs = nn._get_model_outputs(
                features=features,
                descriptors=descriptors,
                mode=tf_estimator.ModeKeys.EVAL,
                verbose=False)
            prediction = dict(
                energy=nn._get_internal_energy_op(outputs, features))

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                energy = float(sess.run(prediction["energy"]))

            assert_almost_equal(energy,
                                lammps.get_potential_energy(atoms), delta=1e-6)

    def test_eam_sutton90(self):
        """
        Test the total energy calculation of `EamAlloyNN` with `AgSutton90`
        using `EAMTransformer`.
        """
        rc = 11.999
        lammps = self.get_lammps_calculator()

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
                energy = float(sess.run(prediction["energy"],
                                        feed_dict=clf.get_feed_dict(atoms)))

        atoms.calc = lammps
        assert_almost_equal(energy,
                            lammps.get_potential_energy(atoms), delta=1e-6)


if __name__ == "__main__":
    nose.main()
