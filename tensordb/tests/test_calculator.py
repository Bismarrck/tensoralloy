#!coding=utf-8

import nose
import unittest
from pathlib import Path
from ase.build import bulk
from nose.tools import assert_almost_equal, assert_equal, assert_true
from tensordb.calculator import VaspAgingCalculator


class TestVaspAgingCalculator(unittest.TestCase):

    def setUp(self):
        self.root = Path(__file__).parent / 'experiment'
        self.config = self.root / 'config.toml'

    def test_may_modify_atoms(self):
        calc = VaspAgingCalculator(self.root, self.config)
        atoms = bulk('Cu', cubic=True) * [2, 2, 2]
        calc.may_modify_atoms(atoms)


if __name__ == '__main__':
    a = TestVaspAgingCalculator()
    a.setUp()
    a.test_may_modify_atoms()
