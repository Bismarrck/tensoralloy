# -*- coding: utf-8 -*-
import numpy as np
from ase.neighborlist import NeighborList
from tensordb.calculator.calculator import VaspCalculator
from tensordb.utils import getitem


class VaspNonEquilibriumCalculator(VaspCalculator):
    """
    This calculator is used to create non-equilibrium structures by randomly moving
    atoms to their neighboring sites.
    """

    def __init__(self, root, config):
        super().__init__(root, config)
        self.workdir = self.root / "neq"

        params = getitem(self.config, ["neq", ])
        self.dmin = params.get("dmin", 1.2)
        if self.dmin <= 1.0:
            print(f"Warning: The 'dmin' is {self.dmin}, maybe too small!")
        self.nmax = params.get("nmax", 3)
        if self.nmax < 1:
            raise ValueError("The 'nmax' should be larger than 1!")
        self.move_factor = params.get("move_factor", 0.6)
        if self.move_factor <= 0:
            raise ValueError("The 'move_factor' should be larger than 0!")
        if self.move_factor >= 1:
            print(f"Warning: The 'move_factor' should be smaller than 1!")
        self.sampling_interval = params.get("interval", 500)
    
    @property
    def random_seed(self):
        return 1

    def may_modify_atoms(self, atoms):
        size = len(atoms)
        n = min(size // 4, np.random.randint(1, self.nmax + 1))
        if n == 0:
            return None
        
        # Copy the original atoms object
        obj = atoms.copy()
        
        # Select random atoms
        indices = np.random.choice(size, n, replace=False)

        # Get the neighbor list
        cutoffs = np.zeros(size)
        cutoffs[indices] = 5.0
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)

        # For each selected atom, move to its nearest neighbor
        for i in indices:
            nl.update(obj)
            R = obj.positions
            neighbors, offsets = nl.get_neighbors(i)
            # Calculate the displacements
            D = R[neighbors] + offsets @ obj.get_cell() - R[i]
            # Find the nearest neighbor
            d = np.linalg.norm(D, axis=1)
            j = np.argmin(d)
            # Skip the atom if it is too close to the selected atom
            if d[j] < self.dmin:
                continue
            smax = min(self.dmin / d[j], 1.0)
            # Make a trial move
            for s in np.arange(smax * self.move_factor, 0.0, -0.05):
                x = R[i] + D[j] * s
                # Check the distances between the new position and other neighbors
                if np.all(np.linalg.norm(R[i] + D - x, axis=1) >= self.dmin):
                    obj.positions[i] = x
                    break
        return obj

    def create_tasks(self, samplers, **kwargs):
        """
        Create VASP high precision DFT calculation tasks.
        For VaspNonEquilibriumCalculator, the random seed is fixed to 1.
        """
        np.random.seed(self.random_seed)
        return super().create_tasks(samplers, **kwargs)
