#!coding=utf-8
"""
Unit tests of the `AdpNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import nose
import unittest
import shutil

from tensorflow_estimator import estimator as tf_estimator
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from ase import Atoms
from os.path import join, exists
from os import makedirs
from typing import List
from nose.tools import assert_almost_equal

from tensoralloy.test_utils import test_dir, assert_array_almost_equal
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.nn.eam.adp import AdpNN
from tensoralloy.nn.eam.potentials import EamAlloyPotential
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.nn.utils import log_tensor
from tensoralloy.extension.interp.cubic import CubicInterpolator
from tensoralloy.precision import precision_scope
from tensoralloy.io.lammps import LAMMPS_COMMAND, read_adp_setfl

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@unittest.skipUnless(CubicInterpolator.runnable(), "Cubic ops lib is not built")
class AlCuSplineAdp(EamAlloyPotential):
    """
    The Cubic Spline form of the Al-Cu ADP potential.

    References
    ----------
    PHYSICAL REVIEW B 83, 054116 (2011)

    """

    def __init__(self, interval=1):
        """
        Initialization method.
        """
        super(AlCuSplineAdp, self).__init__()

        filename = join(test_dir(), 'lammps', 'AlCu.adp')
        self._adpfl = read_adp_setfl(filename)
        self._name = "AlCuAdp"
        self._interval = interval

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            verbose=False):
        """
        The electron density function.
        """
        with tf.name_scope(f"{self._name}/Rho/{element}"):
            nr = self._adpfl.nr
            x = self._adpfl.rho[element].x[0:nr:self._interval]
            y = self._adpfl.rho[element].y[0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            rho = f.run(tf.reshape(r, (-1,), name='r/flat'))
            rho = tf.reshape(rho, shape, name='rho')
            if verbose:
                log_tensor(rho)
            return rho

    def embed(self,
              rho: tf.Tensor,
              element: str,
              variable_scope: str,
              verbose=False):
        """
        The embedding function.
        """
        with tf.name_scope(f"{self._name}/Embed/{element}"):
            nrho = self._adpfl.nrho
            x = self._adpfl.embed[element].x[0:nrho:self._interval]
            y = self._adpfl.embed[element].y[0:nrho:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(rho, name='shape')
            frho = f.run(tf.reshape(rho, (-1,), name='rho/flat'))
            frho = tf.reshape(frho, shape, name='frho')
            if verbose:
                log_tensor(frho)
            return frho

    def phi(self,
            r: tf.Tensor,
            kbody_term: str,
            variable_scope: str,
            verbose=False):
        """
        The pairwise interaction function.
        """
        with tf.name_scope(f"{self._name}/Phi/{kbody_term}"):
            nr = self._adpfl.nr
            x = self._adpfl.phi[kbody_term].x[0:nr:self._interval]
            y = self._adpfl.phi[kbody_term].y[0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            phi = f.run(tf.reshape(r, (-1,), name='r/flat'))
            phi = tf.reshape(phi, shape, name='phi')
            if verbose:
                log_tensor(phi)
            return phi

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               verbose=False):
        """
        The dipole function.
        """
        with tf.name_scope(f"{self._name}/Dipole/{kbody_term}"):
            nr = self._adpfl.nr
            x = self._adpfl.dipole[kbody_term].x[0:nr:self._interval]
            y = self._adpfl.dipole[kbody_term].y[0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            dipole = f.run(tf.reshape(r, (-1,), name='r/flat'))
            dipole = tf.reshape(dipole, shape, name='dipole')
            if verbose:
                log_tensor(dipole)
            return dipole

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   verbose=False):
        """
        The quadrupole function.
        """
        with tf.name_scope(f"{self._name}/Quadrupole/{kbody_term}"):
            nr = self._adpfl.nr
            x = self._adpfl.quadrupole[kbody_term].x[0:nr:self._interval]
            y = self._adpfl.quadrupole[kbody_term].y[0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            quadrupole = f.run(tf.reshape(r, (-1,), name='r/flat'))
            quadrupole = tf.reshape(quadrupole, shape, name='quadrupole')
            if verbose:
                log_tensor(quadrupole)
            return quadrupole


@unittest.skipUnless(CubicInterpolator.runnable(), "Cubic ops lib is not built")
class AlCuAdpTest(unittest.TestCase):
    """
    Test cubic spline functions using Al-Cu adp.
    """

    def setUp(self):
        """
        The setup function.
        """
        self.work_dir = join(test_dir(), 'lammps', 'adp')
        if not exists(self.work_dir):
            makedirs(self.work_dir)

    def tearDown(self) -> None:
        """
        The cleanup function.
        """
        if exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    def get_lammps_calculator(self, elements: List[str], use_official=False):
        """
        Return a LAMMPS calculator.
        """
        if use_official:
            pot_file = join(test_dir(), 'lammps', 'AlCu.adp')
        else:
            pot_file = join(self.work_dir, 'AlCu.adp')
        work_dir = join(self.work_dir, ''.join(elements))
        if not exists(work_dir):
            makedirs(work_dir)

        return LAMMPS(files=[pot_file],
                      binary_dump=False,
                      write_velocities=False,
                      tmp_dir=work_dir,
                      keep_tmp_files=False,
                      keep_alive=False,
                      no_data_file=False,
                      pair_style="adp",
                      command=LAMMPS_COMMAND,
                      pair_coeff=[f"* * AlCu.adp {' '.join(elements)}"])

    def _run(self, atoms: Atoms, rc=7.5):
        """
        Run a test.
        """
        with precision_scope("high"):
            with tf.Graph().as_default():
                available_potentials['adp/AlCu'] = AlCuSplineAdp
                elements = sorted(list(set(atoms.get_chemical_symbols())))
                clf = UniversalTransformer(rcut=rc, elements=elements)
                nn = AdpNN(elements,
                           custom_potentials="adp/AlCu",
                           export_properties=['energy', 'forces', 'stress'])
                nn.attach_transformer(clf)
                nn.export_to_setfl(
                    join(self.work_dir, 'AlCu.adp'),
                    nr=10000, dr=rc / 1e4, nrho=10000, drho=0.005)

                predictions = nn.build(
                    features=clf.get_constant_features(atoms),
                    mode=tf_estimator.ModeKeys.PREDICT,
                    verbose=True)

                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    results = sess.run(predictions)

        mishin = self.get_lammps_calculator(elements, use_official=True)
        mishin.calculate(atoms)
        export = self.get_lammps_calculator(elements, use_official=False)
        export.calculate(atoms)

        assert_almost_equal(mishin.get_potential_energy(atoms),
                            export.get_potential_energy(atoms), delta=1e-5)
        assert_array_almost_equal(mishin.get_stress(atoms),
                                  export.get_stress(atoms), delta=1e-5)
        assert_almost_equal(results["energy"],
                            mishin.get_potential_energy(atoms),
                            delta=1e-5)
        assert_array_almost_equal(results["stress"],
                                  mishin.get_stress(atoms),
                                  delta=1e-5)

    def test_alloy(self):
        """
        Test Al-Cu alloy ADP calculation.
        """
        atoms = bulk('Al', crystalstructure='fcc', cubic=True)
        atoms.set_chemical_symbols(['Al', 'Cu', 'Al', 'Al'])
        self._run(atoms, rc=7.5)

    def test_bulk(self):
        """
        Test Al bulk ADP calculation.
        """
        atoms = bulk('Al', crystalstructure='fcc', cubic=True)
        self._run(atoms, rc=6.5)


if __name__ == "__main__":
    nose.run()
