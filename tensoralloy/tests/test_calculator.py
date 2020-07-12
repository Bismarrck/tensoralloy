# coding=utf-8
"""
This module defines unit tests of `TensorAlloyCalculator`.
"""
from __future__ import print_function, absolute_import

import numpy as np
import nose
import shutil
import os
import unittest

from unittest import skipUnless
from datetime import datetime
from nose.tools import assert_almost_equal, assert_equal
from nose.tools import assert_list_equal, assert_is_not_none, assert_less
from os.path import join, exists
from ase.db import connect
from ase.calculators.lammpsrun import LAMMPS
from ase.build import bulk
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    from pymatgen import Lattice, Structure
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.elasticity.elastic import ElasticTensor

except ImportError:
    is_pymatgen_avail = False

    ElasticTensor = None
    Lattice = None
    Structure = None
    SlabGenerator = None
    AseAtomsAdaptor = None
    SpacegroupAnalyzer = None

else:
    is_pymatgen_avail = True

from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.transformer import EAMTransformer
from tensoralloy.nn import EamAlloyNN
from tensoralloy.utils import Defaults
from tensoralloy.io.lammps import LAMMPS_COMMAND
from tensoralloy.test_utils import test_dir, datasets_dir
from tensoralloy.test_utils import assert_array_almost_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'



class ModelTimeStampTest(unittest.TestCase):

    def setUp(self):
        """
        Setup this test case.
        """
        self.pb_file = join(test_dir(), 'Ni.test.pb')

    def test_get_model_timestamp(self):
        """
        Test the method `get_model_timestamp`.
        """
        nn = EamAlloyNN(['Ni'], 'zjw04xc',
                        export_properties=['energy', 'forces', 'stress',
                                           'hessian'])
        nn.attach_transformer(EAMTransformer(6.0, ['Ni']))
        nn.export(self.pb_file, keep_tmp_files=False)

        calc = TensorAlloyCalculator(self.pb_file)
        timestamp = calc.get_model_timestamp()
        assert_is_not_none(timestamp)

        encoded = datetime.fromisoformat(timestamp)
        now = datetime.now()

        assert_less((now - encoded).total_seconds(), 30)

    def tearDown(self):
        """
        The cleanup function for this test case.
        """
        if exists(self.pb_file):
            os.remove(self.pb_file)


def test_calculator_with_qm7():
    """
    Test total energy calculation of `TensorAlloyCalculator` using the QM7
    dataset.
    """
    graph_path = join(test_dir(), 'checkpoints', 'qm7-k2', 'qm7.pb')
    sqlite3_file = join(datasets_dir(), 'qm7.db')

    calc = TensorAlloyCalculator(graph_path)

    db = connect(sqlite3_file)
    _, tests = train_test_split(np.arange(1, 1 + len(db)),
                                random_state=Defaults.seed, test_size=1000)

    y_true = []
    y_pred = []
    for index in tests:
        atoms = db.get_atoms(f'id={index}')
        y_true.append(atoms.get_total_energy())
        y_pred.append(calc.get_potential_energy(atoms))

    y_mae = mean_absolute_error(y_true, y_pred)
    assert_almost_equal(y_mae, 0.36505362, delta=1e-6)


@skipUnless(os.environ.get('TEST_ELASTIC') and is_pymatgen_avail,
            "The flag 'TEST_ELASTIC' is not set or PyMatgen not available")
def test_elastic_constant_tensor():
    """
    Test elastic properties calculation of `TensorAlloyCalculator`.
    """
    graph_path = join(test_dir(), 'models', "Ni.zhou04.elastic.pb")
    calc = TensorAlloyCalculator(graph_path)

    cubic = bulk("Ni", cubic=True)
    tensor = calc.get_elastic_constant_tensor(cubic)
    tensor = ElasticTensor.from_voigt(tensor)

    assert_almost_equal(tensor.voigt[0, 0], 246.61, delta=0.01)
    assert_almost_equal(tensor.voigt[1, 1], 246.61, delta=0.01)
    assert_almost_equal(tensor.voigt[2, 2], 246.61, delta=0.01)
    assert_almost_equal(tensor.voigt[0, 1], 147.15, delta=0.01)
    assert_almost_equal(tensor.voigt[0, 2], 147.15, delta=0.01)
    assert_almost_equal(tensor.voigt[3, 3], 124.72, delta=0.01)
    assert_almost_equal(tensor.voigt[4, 4], 124.72, delta=0.01)
    assert_almost_equal(tensor.voigt[5, 5], 124.72, delta=0.01)
    assert_almost_equal(tensor.homogeneous_poisson, 0.2936, delta=0.0001)
    assert_almost_equal(tensor.k_vrh, 180.31, delta=0.01)
    assert_almost_equal(tensor.g_vrh, 86.261, delta=0.01)

    primitive = bulk("Ni", cubic=False)
    structure = AseAtomsAdaptor.get_structure(primitive)
    analyzer = SpacegroupAnalyzer(structure)
    primitive = AseAtomsAdaptor.get_atoms(
        analyzer.get_conventional_standard_structure())

    tensor_p = calc.get_elastic_constant_tensor(primitive)

    assert_array_almost_equal(tensor, tensor_p, delta=1e-6)


@skipUnless(exists(LAMMPS_COMMAND), f"LAMMPS not found!")
class SurfaceSlabTest(unittest.TestCase):
    """
    Test total energy calculation of `TensorAlloyCalculator` with EAM method for
    Ni surface slabs.
    """

    def setUp(self):
        """
        The setup function.
        """
        self.tmp_dir = join(test_dir(), "tmp")
        if not exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def get_lammps_calculator(self):
        """
        Return a LAMMPS calculator for Ag.
        """
        eam_file = join(test_dir(absolute=True), 'lammps', 'zjw04_Ni.alloy.eam')

        return LAMMPS(files=[eam_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=self.tmp_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="eam/alloy",
                      command=LAMMPS_COMMAND,
                      pair_coeff=['* * zjw04_Ni.alloy.eam Ni'])

    def tearDown(self):
        """
        The cleanup function.
        """
        if exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @skipUnless(is_pymatgen_avail, "PyMatgen is not available")
    def test_calculator_with_eam_slab(self):
        """
        Test total energy calculation of `TensorAlloyCalculator` with EAM method for
        Ni surface slabs.
        """
        lammps = self.get_lammps_calculator()

        # Initialize a calculator
        graph_model_path = join(test_dir(), 'models', 'Ni.zhou04.pb')
        clf = TensorAlloyCalculator(graph_model_path)

        # Create several slabs
        lattice = Lattice.cubic(3.524)
        coords = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                             [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        base_structure = Structure(
            lattice, ['Ni', 'Ni', 'Ni', 'Ni'],
            coords=coords)

        miller_indices = [
            [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0],
            [2, 1, 1], [2, 2, 1], [3, 1, 0], [3, 1, 1],
            [3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 3, 1],
            [3, 3, 2], [4, 1, 1], [4, 2, 1], [4, 3, 1],
            [4, 3, 2], [4, 3, 3], [4, 4, 1], [4, 4, 3],
        ]
        structures = {}
        y_true = []
        y_pred = []

        for miller_index in miller_indices:
            key = ''.join(map(str, miller_index))
            slabgen = SlabGenerator(base_structure, miller_index, 10.0, 10.0)
            all_slabs = slabgen.get_slabs()
            slab = all_slabs[0]

            assert_equal(len(all_slabs), 1)
            assert_list_equal(list(slab.miller_index), miller_index)

            slab.make_supercell([2, 2, 1])
            structures[key] = slab

            atoms = AseAtomsAdaptor.get_atoms(slab)
            y_true.append(lammps.get_potential_energy(atoms))
            y_pred.append(clf.get_potential_energy(atoms))

        assert_array_almost_equal(np.asarray(y_true), np.asarray(y_pred), 4e-5)


if __name__ == "__main__":
    nose.run()
