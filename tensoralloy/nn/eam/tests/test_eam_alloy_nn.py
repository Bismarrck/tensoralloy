# coding=utf-8
"""
This module defines unit tests of `EamAlloyNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import os
import shutil
import unittest

from nose.tools import assert_equal, assert_almost_equal
from nose.tools import assert_dict_equal, assert_list_equal, with_setup
from os.path import join, exists
from os import remove
from collections import Counter
from unittest import skipUnless
from ase.calculators.lammpsrun import LAMMPS
from ase.build import bulk
from ase.db import connect
from ase.units import GPa
from ase.io import read

try:
    from pymatgen import Lattice, Structure
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    is_pymatgen_avail = False
    Lattice = None
    Structure = None
    SlabGenerator = None
    AseAtomsAdaptor = None
else:
    is_pymatgen_avail = True

from tensoralloy.utils import ModeKeys
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.test_utils import assert_array_equal, datasets_dir
from tensoralloy.test_utils import assert_array_almost_equal, test_dir, data_dir
from tensoralloy.utils import Defaults
from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.io.lammps import LAMMPS_COMMAND


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_hidden_sizes():
    """
    Test setting hidden layer sizes of `EamAlloyNN`.
    """
    custom_potentials = {
        'AlCu': {'phi': 'zjw04'},
        'Al': {'rho': 'zjw04'},
        'CuCu': {'phi': 'zjw04'},
    }
    nn = EamAlloyNN(elements=['Al', 'Cu'],
                    hidden_sizes={'AlAl': {'phi': [128, 64]}},
                    custom_potentials=custom_potentials)
    assert_dict_equal(nn.potentials,
                      {'AlAl': {'phi': 'nn'},
                       'CuCu': {'phi': 'zjw04'},
                       'AlCu': {'phi': 'zjw04'},
                       'Al': {'rho': 'zjw04', 'embed': 'nn'},
                       'Cu': {'rho': 'nn', 'embed': 'nn'}})
    assert_equal(nn.hidden_sizes,
                 {'AlAl': {'phi': [128, 64]},
                  'CuCu': {'phi': Defaults.hidden_sizes},
                  'AlCu': {'phi': Defaults.hidden_sizes},
                  'Al': {'rho': Defaults.hidden_sizes,
                         'embed': Defaults.hidden_sizes},
                  'Cu': {'rho': Defaults.hidden_sizes,
                         'embed': Defaults.hidden_sizes}})


def test_custom_potentials():
    """
    Test setting layers of `EamAlloyNN`.
    """
    with tf.Graph().as_default():
        custom_potentials = {
            'AlCu': {'phi': 'zjw04'},
            'Al': {'rho': 'zjw04'},
            'CuCu': {'phi': 'zjw04'},
        }
        nn = EamAlloyNN(elements=['Al', 'Cu'],
                        custom_potentials=custom_potentials)
        assert_dict_equal(nn.potentials,
                          {'AlAl': {'phi': 'nn'},
                           'CuCu': {'phi': 'zjw04'},
                           'AlCu': {'phi': 'zjw04'},
                           'Al': {'rho': 'zjw04', 'embed': 'nn'},
                           'Cu': {'rho': 'nn', 'embed': 'nn'}})


def test_as_dict():
    """
    Test the inherited `EamAlloyNN.as_dict().
    """
    custom_potentials = {
        'AlCu': {'phi': 'zjw04'},
        'Al': {'rho': 'zjw04'},
        'CuCu': {'phi': 'zjw04'},
    }
    old_nn = EamAlloyNN(elements=['Al', 'Cu'],
                        custom_potentials=custom_potentials)

    d = old_nn.as_dict()
    assert_equal(d.pop('class'), 'EamAlloyNN')

    new_nn = EamAlloyNN(**d)
    assert_dict_equal(new_nn.potentials, old_nn.potentials)
    assert_dict_equal(new_nn.hidden_sizes, old_nn.hidden_sizes)
    assert_list_equal(new_nn.minimize_properties, old_nn.minimize_properties)
    assert_list_equal(new_nn.predict_properties, old_nn.predict_properties)
    assert_list_equal(new_nn.elements, old_nn.elements)
    assert_equal(new_nn._activation, old_nn._activation)


def export_setfl_teardown():
    """
    Remove the generated setfl file.
    """
    for afile in ('AlCu.alloy.eam',
                  'Al.embed.png', 'Cu.embed.png', 'Al.rho.png', 'Cu.rho.png',
                  'AlCu.phi.png', 'AlAl.phi.png', 'CuCu.phi.png'):
        if exists(afile):
            remove(afile)


@with_setup(teardown=export_setfl_teardown)
def test_export_setfl():
    """
    Test exporting eam/alloy model of AlCuZJW04 to a setfl file.
    """
    nn = EamAlloyNN(
        elements=['Al', 'Cu'],
        custom_potentials={'Al': {'rho': 'zjw04', 'embed': 'zjw04'},
                           'Cu': {'rho': 'zjw04', 'embed': 'zjw04'},
                           'AlAl': {'phi': 'zjw04'},
                           'AlCu': {'phi': 'zjw04'},
                           'CuCu': {'phi': 'zjw04'}})
    nn.export_to_setfl('AlCu.alloy.eam',
                       nr=2000, dr=0.003, nrho=2000, drho=0.05)

    with open('AlCu.alloy.eam') as fp:
        out = []
        for i, line in enumerate(fp):
            if i < 6 or i == 4006:
                continue
            out.append(float(line.strip()))
    with open(join(test_dir(), 'lammps', 'Zhou_AlCu.alloy.eam')) as fp:
        ref = []
        for i, line in enumerate(fp):
            if i < 6 or i == 4006:
                continue
            ref.append(float(line.strip()))
    assert_array_equal(np.asarray(out), np.asarray(ref))


def voigt_to_full(tensor):
    """
    A helper function converting the Voigt tensor to a full 3x3 tensor.
    """
    assert len(tensor) == 6
    xx, yy, zz, yz, xz, xy = tensor
    return np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]])


class NiMoAlloyTest(unittest.TestCase):
    """
    Test energy, force and stress calculations of Ni3Mo and Ni4Mo alloys.
    """

    def setUp(self):
        """
        Setup this test.
        """
        work_dir = join(test_dir(), 'zjw04')
        if not exists(work_dir):
            os.makedirs(work_dir)

        self.work_dir = work_dir
        self.pb_file = join(work_dir, 'NiMo.pb')
        self.lammps = self.get_lammps_calculator()

        elements = ['Mo', 'Ni']
        nn = EamAlloyNN(elements, 'zjw04',
                        export_properties=['energy', 'forces', 'stress'])
        clf = UniversalTransformer(elements=elements, rcut=6.5)
        nn.attach_transformer(clf)
        nn.export(self.pb_file, keep_tmp_files=True)

    def get_lammps_calculator(self):
        """
        Return a LAMMPS calculator for Ag.
        """
        eam_file = join(test_dir(True), 'lammps', 'MoNi_Zhou04.eam.alloy')
        return LAMMPS(files=[eam_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=self.work_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="eam/alloy",
                      command=LAMMPS_COMMAND,
                      pair_coeff=['* * MoNi_Zhou04.eam.alloy Mo Ni'])

    def test(self):
        """
        The main tests.
        """
        calc = TensorAlloyCalculator(self.pb_file)

        crysts_dir = join(data_dir(), 'crystals')
        files = {
            'Ni3Mo': 'Ni3Mo_mp-11506_conventional_standard.cif',
            'Ni4Mo': 'Ni4Mo_mp-11507_conventional_standard.cif'
        }
        eps = 1e-5

        for name, cif in files.items():
            cryst = read(join(crysts_dir, cif))

            lmp = self.get_lammps_calculator()
            lmp.calculate(cryst)

            energy = lmp.get_potential_energy(cryst)
            forces = lmp.get_forces(cryst)
            stress = lmp.get_stress(cryst)

            calc.calculate(cryst)

            assert_almost_equal(energy, calc.get_potential_energy(cryst),
                                delta=eps, msg=f'{name}.energy')
            assert_array_almost_equal(forces, calc.get_forces(cryst),
                                      delta=eps, msg=f'{name}.forces')
            assert_array_almost_equal(stress, calc.get_stress(cryst, True),
                                      delta=eps, msg=f'{name}.stress')

    def tearDown(self):
        """
        Delete the tmp dir.
        """
        if exists(self.lammps.tmp_dir):
            shutil.rmtree(self.lammps.tmp_dir, ignore_errors=True)
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir)


class BulkStressOpTest(unittest.TestCase):
    """
    Test the stress calculation with a bulk Al-Cu using EAM with Zjw04
    potential.
    """

    def setUp(self):
        """
        Setup this test.
        """
        work_dir = join(test_dir(True), 'lammps', 'zjw04')
        if not exists(work_dir):
            os.makedirs(work_dir)

        self.work_dir = work_dir
        self.lammps = self.get_lammps_calculator()

    def get_lammps_calculator(self):
        """
        Return a LAMMPS calculator for Ag.
        """
        eam_file = join(test_dir(True), 'lammps', 'Zhou_AlCu.alloy.eam')
        return LAMMPS(files=[eam_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=self.work_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="eam/alloy",
                      command=LAMMPS_COMMAND,
                      pair_coeff=['* * Zhou_AlCu.alloy.eam Al Cu'])

    def tearDown(self):
        """
        Delete the tmp dir.
        """
        if exists(self.lammps.tmp_dir):
            shutil.rmtree(self.lammps.tmp_dir, ignore_errors=True)

    @skipUnless(exists(LAMMPS_COMMAND), 'LAMMPS not found!')
    def test_eam_alloy_zjw04_bulk(self):
        """
        Test the energy, forces and stress results of `EamAlloyNN` with `Zjw04`
        potentials for bulk Al-Cu.
        """
        rc = 6.0

        atoms = bulk('Cu') * [2, 2, 2]
        symbols = atoms.get_chemical_symbols()
        symbols[0: 2] = ['Al', 'Al']
        atoms.set_chemical_symbols(symbols)
        elements = sorted(set(symbols))

        with tf.Graph().as_default():
            clf = UniversalTransformer(elements=elements, rcut=rc)
            nn = EamAlloyNN(elements=elements,
                            custom_potentials="zjw04",
                            export_properties=['energy', 'forces', 'stress'])
            nn.attach_transformer(clf)
            predictions = nn.build(
                features=clf.get_placeholder_features(),
                mode=ModeKeys.PREDICT,
                verbose=True)

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                result = sess.run(predictions,
                                  feed_dict=clf.get_feed_dict(atoms))

        atoms.calc = self.lammps
        self.lammps.calculate(atoms)

        assert_almost_equal(result['energy'],
                            self.lammps.get_potential_energy(atoms), delta=1e-6)
        assert_array_almost_equal(result['forces'],
                                  self.lammps.get_forces(atoms), delta=1e-9)

        lmp_voigt_stress = self.lammps.get_stress(atoms)
        stress = voigt_to_full(lmp_voigt_stress)
        assert_array_almost_equal(voigt_to_full(result['stress']),
                                  stress,
                                  delta=1e-5)


def test_batch_stress():
    """
    Test the batch implementation of `_get_reduced_full_stress_tensor`.
    """
    db = connect(join(datasets_dir(), 'snap-Ni.db'))
    atoms = db.get_atoms('id=1')

    volume = atoms.get_volume()
    rc = 6.0
    elements = ['Ni']
    nn = EamAlloyNN(elements,
                    custom_potentials={'Ni': {'rho': 'zjw04', 'embed': 'zjw04'},
                                       'NiNi': {'phi': 'zjw04'}},
                    minimize_properties=['energy', 'stress'],
                    export_properties=['energy', 'forces', 'stress'])

    with tf.Graph().as_default():
        clf = UniversalTransformer(elements=elements, rcut=rc)
        nn.attach_transformer(clf)
        predictions = nn.build(features=clf.get_placeholder_features(),
                               mode=ModeKeys.PREDICT,
                               verbose=False)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            s_true = sess.run(predictions["stress"],
                              feed_dict=clf.get_feed_dict(atoms)) * volume

    with tf.Graph().as_default():
        size = find_neighbor_size_of_atoms(atoms, rc=rc)
        max_occurs = Counter(atoms.get_chemical_symbols())
        clf = BatchUniversalTransformer(
            rcut=rc, max_occurs=max_occurs, nij_max=size.nij, nnl_max=size.nnl,
            batch_size=1, use_forces=True, use_stress=True)
        protobuf = tf.convert_to_tensor(clf.encode(atoms).SerializeToString())
        example = clf.decode_protobuf(protobuf)

        batch = dict()
        for key, tensor in example.items():
            batch[key] = tf.expand_dims(
                tensor, axis=0, name=tensor.op.name + '/batch')

        descriptors = clf.get_descriptors(batch)
        features = dict(positions=batch["positions"],
                        n_atoms_vap=batch["n_atoms_vap"],
                        cell=batch["cell"],
                        atom_masks=batch["atom_masks"],
                        volume=batch["volume"])

        outputs = nn._get_model_outputs(
            features=features,
            descriptors=descriptors,
            mode=ModeKeys.EVAL,
            verbose=False)
        ops = nn._get_energy_ops(outputs, features, verbose=False)
        forces = nn._get_forces_op(ops.energy, batch["positions"],
                                   verbose=False)
        stress, virial, _ = nn._get_stress_ops(
            energy=ops.energy.total,
            cell=batch["cell"],
            volume=batch["volume"],
            positions=batch["positions"],
            forces=forces,
            verbose=False)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            y1, y2 = sess.run([stress, virial])
            y2 = y2[0][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]

        assert_array_almost_equal(s_true, y1 * volume, delta=1e-8)
        assert_array_almost_equal(s_true, y2, delta=1e-8)


@skipUnless(is_pymatgen_avail, "PyMatgen is not available")
class Zjw04SurfaceStressTest(unittest.TestCase):
    """
    Test the stress results of built-in Zjw04 with the potential file generated
    by Zhou's fortran code for a Mo surface.
    """

    def setUp(self):
        """
        Setup this test.
        """
        self.work_dir = join(test_dir(absolute=True), 'lammps', 'slabs')
        if not exists(self.work_dir):
            os.makedirs(self.work_dir)

        lattice = Lattice.cubic(3.52)
        coords = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                             [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        structure = Structure(lattice, ['Ni', 'Ni', 'Ni', 'Ni'],
                              coords=coords)

        slabs = {}
        for miller_index in ((1, 0, 0), (2, 1, 1)):
            key = "".join(map(str, miller_index))
            slabgen = SlabGenerator(structure, miller_index, 10.0, 10.0)
            all_slabs = slabgen.get_slabs()
            slab = all_slabs[0]
            slab.make_supercell((2, 2, 1))
            slabs[key] = slab
        self.slabs = slabs

    def get_lammps_calculator(self, tag):
        """
        Return a LAMMPS calculator for Ag.
        """
        tmp_dir = join(self.work_dir, tag)
        if not exists(tmp_dir):
            os.makedirs(tmp_dir)

        potential_file = join(test_dir(True), 'lammps', 'zjw04_Ni.alloy.eam')
        return LAMMPS(files=[potential_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=tmp_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="eam/alloy",
                      command=LAMMPS_COMMAND,
                      pair_coeff=['* * zjw04_Ni.alloy.eam Ni'])

    @skipUnless(exists(LAMMPS_COMMAND), 'LAMMPS not found!')
    def test_surface_slab(self):
        """
        Test the stress calculations for surface slabs.
        """
        np.set_printoptions(precision=6, suppress=True)

        rc = 6.0
        elements = ['Ni']
        pb_file = join(self.work_dir, 'Ni.zjw04.pb')

        with tf.Graph().as_default():

            nn = EamAlloyNN(elements, custom_potentials='zjw04',
                            export_properties=['energy', 'forces', 'stress'])
            clf = UniversalTransformer(elements=elements, rcut=rc)
            nn.attach_transformer(clf)

            predictions = nn.build(
                features=clf.get_placeholder_features(),
                mode=ModeKeys.PREDICT,
                verbose=True)

            nn.export(pb_file)

            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                direct_results = {}
                for key, slab in self.slabs.items():
                    atoms = AseAtomsAdaptor.get_atoms(slab)
                    direct_results[key] = sess.run(
                        predictions,
                        feed_dict=clf.get_feed_dict(atoms))['stress'] / GPa

        calc = TensorAlloyCalculator(pb_file)
        pb_results = {}
        for key, slab in self.slabs.items():
            atoms = AseAtomsAdaptor.get_atoms(slab)
            pb_results[key] = calc.get_stress(atoms) / GPa

        lmp_results = {}
        for key, slab in self.slabs.items():
            lmp = self.get_lammps_calculator(key)
            atoms = AseAtomsAdaptor.get_atoms(slab)
            atoms.calc = lmp
            lmp.calculate(atoms)
            lmp_stress = lmp.get_stress(atoms) / GPa
            lmp_results[key] = lmp_stress

        eps = 1e-6
        for key in self.slabs:
            assert_array_almost_equal(pb_results[key], direct_results[key], eps)
            assert_array_almost_equal(lmp_results[key], pb_results[key], eps)

    def tearDown(self):
        """
        Cleanup this test.
        """
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)


if __name__ == "__main__":
    nose.run()
