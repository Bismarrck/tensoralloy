#!coding=utf-8

import nose
import numpy as np
import unittest
from collections import Counter
from pathlib import Path
from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list
from tensordb.aging import VaspAgingCalculator
from tensordb.aging import FibnonacciSphereHeliumBubbleInjector


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
        new_atoms = injector.inject(atoms, 0, 3, 5)
        ilist, jlist, dlist = neighbor_list('ijd', new_atoms, cutoff=cutoff)
        chemical_symbols = new_atoms.get_chemical_symbols()
        mindist = Counter()
        for i, j, d in zip(ilist, jlist, dlist):
            key = "".join(sorted([chemical_symbols[i], chemical_symbols[j]]))
            mindist[key] = min(mindist[key], d) if key in mindist else d
        for key, value in mindist.items():
            print(key, value)
        write('fib.POSCAR', new_atoms, vasp5=True)


if __name__ == '__main__':
    a = TestVaspAgingCalculator()
    a.setUp()
    print("Fibonacci Sphere Helium Bubble Injector: ")
    a.test_fibonacci_sphere_injector()
