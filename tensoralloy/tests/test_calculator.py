# coding=utf-8
"""
This module defines unit tests of `TensorAlloyCalculator`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import shutil
import os
import unittest

from datetime import datetime
from nose.tools import assert_almost_equal, with_setup, assert_equal
from nose.tools import assert_list_equal, assert_is_not_none, assert_less
from os.path import join, exists
from ase.db import connect
from ase.calculators.lammpsrun import LAMMPS
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pymatgen import Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.transformer import EAMTransformer
from tensoralloy.nn import EamAlloyNN
from tensoralloy.utils import Defaults
from tensoralloy.test_utils import test_dir, datasets_dir
from tensoralloy.test_utils import assert_array_almost_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'



class ModelTimeStampTest(unittest.TestCase):

    def setUp(self):
        """
        Setup this test case.
        """
        self.pb_file = 'Ni.test.pb'

    def test_get_model_timestamp(self):
        """
        Test the method `get_model_timestamp`.
        """
        nn = EamAlloyNN(['Ni'], 'zjw04xc',
                        export_properties=['energy', 'forces', 'stress', 'hessian'])
        nn.attach_transformer(EAMTransformer(6.0, ['Ni']))
        nn.export(self.pb_file)

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
    assert_almost_equal(y_mae, 0.2001322, delta=1e-7)


_eam_slab_tmpdir = join(test_dir(), 'tmp')

if not exists(_eam_slab_tmpdir):
    os.makedirs(_eam_slab_tmpdir)


# Setup the environment for `LAMMPS`
if 'LAMMPS_COMMAND' not in os.environ:
    LAMMPS_COMMAND = '/usr/local/bin/lmp_serial'
    os.environ['LAMMPS_COMMAND'] = LAMMPS_COMMAND
else:
    LAMMPS_COMMAND = os.environ['LAMMPS_COMMAND']


def get_lammps_calculator():
    """
    Return a LAMMPS calculator for Ag.
    """
    eam_file = join(test_dir(absolute=True), 'lammps', 'zjw04_Ni.alloy.eam')
    parameters = {'pair_style': 'eam/alloy',
                  'pair_coeff': ['* * zjw04_Ni.alloy.eam Ni']}
    work_dir = join(_eam_slab_tmpdir, 'zjw04')
    if not exists(work_dir):
        os.makedirs(work_dir)

    return LAMMPS(files=[eam_file], parameters=parameters, tmp_dir=work_dir,
                  keep_tmp_files=True, keep_alive=False, no_data_file=False)


lammps = get_lammps_calculator()


def teardown_eam_slab():
    """
    The cleanup function for `test_calculator_with_eam_slab`.
    """
    if exists(_eam_slab_tmpdir):
        shutil.rmtree(_eam_slab_tmpdir)
    if exists(lammps.tmp_dir):
        shutil.rmtree(lammps.tmp_dir)


@with_setup(teardown=teardown_eam_slab)
def test_calculator_with_eam_slab():
    """
    Test total energy calculation of `TensorAlloyCalculator` with EAM method for
    Ni surface slabs.
    """
    graph_model_path = join(_eam_slab_tmpdir, 'zjw04.pb')

    # Export the zjw04 pb model file
    with tf.Graph().as_default():
        rc = 6.0
        elements = ['Ni']
        clf = EAMTransformer(rc=rc, elements=elements)
        nn = EamAlloyNN(elements=elements,
                        custom_potentials={
                            "Ni": {"rho": "zjw04", "embed": "zjw04"},
                            "NiNi": {"phi": "zjw04"}},
                        export_properties=['energy', 'forces', 'stress'])
        nn.attach_transformer(clf)
        nn.export(graph_model_path, keep_tmp_files=True)

    tf.reset_default_graph()

    # Initialize a calculator
    clf = TensorAlloyCalculator(graph_model_path)

    # Create several slabs
    lattice = Lattice.cubic(3.524)
    base_structure = Structure(
        lattice, ['Ni', 'Ni', 'Ni', 'Ni'],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])

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

    assert_array_almost_equal(np.asarray(y_true), np.asarray(y_pred), 1e-5)


if __name__ == "__main__":
    nose.run()
