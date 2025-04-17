#!coding=utf-8

import toml
import nose
import numpy as np
import unittest
from collections import Counter
from pathlib import Path
from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list
from tensordb.calculator.microstructure.aging import VaspAgingCalculator
from tensordb.calculator.microstructure.aging import FibnonacciSphereHeliumBubbleInjector
from tensordb.calculator.microstructure.neq import VaspNonEquilibriumCalculator


class TestVaspAgingCalculator(unittest.TestCase):

    def setUp(self):
        self.root = Path(__file__).parent / 'experiment'
        self.config = self.root / 'config.toml'

    def test_may_modify_atoms(self):
        calc = VaspAgingCalculator(self.root, self.config)
        atoms = bulk('Cu', cubic=True) * [2, 2, 2]
        calc.may_modify_atoms(atoms)
    
    def test_fibonacci_sphere_injector(self):
        cutoff = 4.0
        atoms = bulk('Cu', cubic=True) * [2, 2, 2]
        atoms.positions += np.random.rand(*atoms.positions.shape) * 0.1

        injector = FibnonacciSphereHeliumBubbleInjector(cutoff=cutoff)
        new_atoms = injector.inject(atoms, 0, 1, 2)
        ilist, jlist, dlist = neighbor_list('ijd', new_atoms, cutoff=cutoff)
        chemical_symbols = new_atoms.get_chemical_symbols()
        mindist = Counter()
        for i, j, d in zip(ilist, jlist, dlist):
            key = "".join(sorted([chemical_symbols[i], chemical_symbols[j]]))
            mindist[key] = min(mindist[key], d) if key in mindist else d
        for key, value in mindist.items():
            print(key, value)


class TestVaspNonEquilibriumCalculator(unittest.TestCase):

    def setUp(self):
        self.root = Path(__file__).parent / 'experiment'
        with open(self.root / 'config.toml') as f:
            self.config = toml.load(f)
    
    def test_move_1(self):
        self.config['non_equilibrium']['nmax'] = 1
        calc = VaspNonEquilibriumCalculator(self.root, self.config)
        atoms = bulk('Cu', cubic=True) * [2, 2, 2]
        obj = calc.may_modify_atoms(atoms)
        dlist = neighbor_list('d', obj, cutoff=4.0)
        write('neq1.POSCAR', obj, format='vasp')
        assert dlist.min() >= 1.2
    
    def test_move_2(self):
        self.config['non_equilibrium']['nmax'] = 2
        calc = VaspNonEquilibriumCalculator(self.root, self.config)
        atoms = bulk('Cu', cubic=True) * [2, 2, 2]
        obj = calc.may_modify_atoms(atoms)
        dlist = neighbor_list('d', obj, cutoff=4.0)
        write('neq2.POSCAR', obj, format='vasp')
        assert dlist.min() >= 1.2

    def test_move_3(self):
        np.random.seed(0)
        self.config['non_equilibrium']['nmax'] = 3
        calc = VaspNonEquilibriumCalculator(self.root, self.config)
        atoms = bulk('Cu', cubic=True) * [2, 2, 2]
        atoms.positions += np.random.rand(*atoms.positions.shape) * 0.4
        obj = calc.may_modify_atoms(atoms)
        dlist = neighbor_list('d', obj, cutoff=4.0)
        write('neq3.POSCAR', obj, format='vasp')
        assert dlist.min() >= 1.2


def main():
    c1 = TestVaspNonEquilibriumCalculator()
    c1.setUp()
    c1.test_move_1()
    c1.test_move_2()
    c1.test_move_3()


if __name__ == '__main__':
    main()
