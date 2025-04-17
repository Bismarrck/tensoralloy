# -*- coding: utf-8 -*-
# This module defines the calculator for creating irradiation-induced aging structures.

import numpy as np
from dataclasses import dataclass
from typing import List
from ase import Atoms, Atom
from ase.neighborlist import NeighborList
from ase.geometry import get_distances, find_mic
from ase.data import covalent_radii, atomic_numbers
from scipy.optimize import minimize
from tensordb.calculator.calculator import VaspCalculator
from tensordb.sampler import BaseSampler
from tensordb.utils import getitem


@dataclass
class Transmutation:
    """
    The dataclass for defining an irradiation-induced transmutation.
    """
    src: str
    dst: str
    prob: float
    nmax: int = 1
    used: int = 0


@dataclass
class HeliumBubbleInjection:
    """
    The dataclass for defining an irradiation-induced helium bubble injection.
    """
    target: str
    max_target_size: int
    max_bubble_size: int
    max_ratio: float = 3.0
    min_ratio: float = 1.0
    cutoff: float = 4.0

    def __post_init__(self):
        assert 1 <= self.max_target_size <= 5
        assert 1 <= self.max_bubble_size <= 10
        assert self.max_ratio <= 4.0
        assert 0.5 <= self.min_ratio <= self.max_ratio

class VaspAgingCalculator(VaspCalculator):
    """
    The VASP calculator for creating irradiation-induced aging structures.
    Two types of microstructures are considered: helium bubble and transmutated element.
    The input block is as follows:

    >>> 
    [aging]
    interval = 100
    cutoff = 4.0
    [aging.transmutation]
    U-Th = {"prob": 0.5, "nmax": 1}
    [aging.helium_bubble]
    U = {"prob": 0.5, "ratio": 2, "nmax": 1, "dist_mean": 1.5, "dist_std": 0.5}

    Explanations: 
    1. Every 1/100 snapshots are used to create aging structures.
    2. Element U to Th
    * probability is 50%
    * at most 1 U atom is transmutated.
    
    3. Element U to He bubble: 
    * probability is 50%
    * the ratio of U to He is 1:1 or 2:1
    * at most 1 U atom is transmutated
    * the distance between He and He is 1.5 A with a standard deviation of 0.5 A

    """
    defaults = {"prob": 0.5, "ratio": 1, "nmax": 1, "dist_mean": 1.5, "dist_std": 0.5,
                "interval": 100, "cutoff": 4.0}
    
    def __init__(self, root, config):
        super().__init__(root, config)
        self.workdir = self.root / "aging"

        params = getitem(self.config, ["aging", ])
        self.sampling_interval = params.get("interval", self.defaults["interval"])
        self.cutoff = params.get("cutoff", self.defaults["cutoff"])

        # Parse the transmutations
        self.transmutations = []
        for key, value in params.get("transmutation", {}).items():
            if not isinstance(value, dict):
                raise ValueError(f"The value of {key} should be a dictionary.")
            src = key.split("-")[0]
            if src not in self.species:
                raise ValueError(
                    f"The source element {src} is not in the species list.")
            dst = key.split("-")[1]
            prob = value.get("prob", self.defaults["prob"])
            nmax = value.get("nmax", self.defaults["nmax"])
            self.transmutations.append(Transmutation(src, dst, prob, nmax=nmax))
        
        # Parse the helium bubble injections
        self.helium_bubble_injection = None
        args = params.get("helium_bubble", {})
        if len(args) > 0:
            self.helium_bubble_injection = HeliumBubbleInjection(
                target=args["target"], 
                max_target_size=args["max_target_size"],
                max_bubble_size=args["max_bubble_size"],
                max_ratio=args["max_ratio"],
                cutoff=args.get("cutoff", self.defaults["cutoff"])
            )

    @property
    def random_seed(self):
        return 1

    def may_inject_helium_bubble(self, atoms: Atoms, indices: np.ndarray):
        """
        The method for injecting a small helium bubble into the structure.
        """
        if self.helium_bubble_injection is None:
            return

        # Initialize the helium bubble injector
        args = self.helium_bubble_injection
        injector = FibnonacciSphereHeliumBubbleInjector(cutoff=args.cutoff)
        
        # Setup the sizes
        n = 0
        while n < 100:
            cluster_size = np.random.randint(1, args.max_target_size + 1)
            bubble_size = np.random.randint(1, args.max_bubble_size + 1)
            ratio = bubble_size / cluster_size
            if args.min_ratio <= ratio <= args.max_ratio:
                break
            n += 1
        else:
            raise ValueError("Cannot find a valid cluster size and bubble size.")

        # Do the injection
        injector.inject(atoms, indices[0], cluster_size, bubble_size, inplace=True)

    def may_modify_atoms(self, atoms: Atoms):
        """
        Create structures representing irradiation-induced aging.
        The input `Atoms` object must have 16 atoms at least.
        """

        # Check the number of atoms
        n = len(atoms)
        if n < 16:
            return None

        # Reset the counter
        for transmutation in self.transmutations:
            transmutation.used = 0
        
        # 1. Copy the input `Atoms` object.
        obj = atoms.copy()
        
        # 2. Shuffle the atoms to modify.
        obj.info["modified"] = np.zeros(n, dtype=bool)
        shuffled_indices = np.random.permutation(n)
        
        # 3. Loop through each helium bubble injection.
        self.may_inject_helium_bubble(obj, shuffled_indices)

        # 4. Loop through each transmutation.
        for transmutation in self.transmutations:
            for i in shuffled_indices:
                if obj.info["modified"][i]:
                    continue
                if obj[i].symbol == transmutation.src:
                    # Check the probability
                    if np.random.uniform(0, 1) <= transmutation.prob:
                        obj[i].symbol = transmutation.dst
                        obj.info["modified"][i] = True
                        # Check the number of atoms used for this transmutation
                        transmutation.used += 1
                        if transmutation.used >= transmutation.nmax:
                            break
        return obj

    def create_tasks(self, samplers: List[BaseSampler], **kwargs):
        """
        Create VASP high precision DFT calculation tasks.
        """
        super().create_tasks(samplers, **kwargs)


class HeliumBubbleInjector:
    """
    The class for injecting helium bubbles into the structure.
    """

    def __init__(self, cutoff=4.0, optimization_steps=100):
        """
        The initialization method.
        """
        self.cutoff = cutoff
        self.optimization_steps = optimization_steps
    
    @staticmethod
    def pbc_aware_centroid(positions, cell, pbc):
        """
        Calculate centroid considering periodic boundaries using MIC
        """
        if not np.any(pbc):  # No PBC in any direction
            return np.mean(positions, axis=0)
        
        # Reference point (first atom in cluster)
        ref_pos = positions[0]
        displacements = positions - ref_pos
        
        # Apply minimum image convention to all displacements
        adjusted_displacements, _ = find_mic(displacements, cell, pbc)
        
        # Reconstruct absolute positions in reference image
        adjusted_positions = ref_pos + adjusted_displacements
        
        return np.mean(adjusted_positions, axis=0)

    def inject(self, atoms: Atoms, center_atom_idx: int, cluster_size: int, 
               bubble_size: int, inplace=False) -> Atoms:
        """
        Inject helium bubbles into 'atoms'.

        Args:
            atoms: ASE Atoms object
            center_atom_idx: Index of the center atom
            cluster_size: Number of atoms in the cluster
            bubble_size: Number of helium atoms to inject
            inplace: Modify the input Atoms object in place

        """
        cluster_indices = find_cluster_neighbor_list_method(
            atoms, center_atom_idx, cluster_size, self.cutoff)
        return self.replace_cluster(atoms, cluster_indices, "He", bubble_size, 
                                    inplace=inplace)

    def replace_cluster(self, 
                        atoms: Atoms, 
                        cluster_indices: list, 
                        new_element: str, 
                        new_atom_count: int, 
                        inplace: bool = False) -> Atoms:
        """
        Replace a cluster of atoms with a new element and optimize the positions.
        """
        raise NotImplementedError
    
    def replace_atoms(self, atoms, old_indices, new_element, new_positions):
        """
        Replace a group of atoms with new atoms in the structure
        """
        del atoms[old_indices]
        new_indices = []
        n = len(atoms)
        for pos in new_positions:
            atoms.append(Atom(new_element, position=pos))
            new_indices.append(n)
            n += 1
        atoms.wrap()
        return new_indices


class FibnonacciSphereHeliumBubbleInjector(HeliumBubbleInjector):
    """
    The class for injecting helium bubbles using the Fibonacci sphere method.
    """

    def replace_cluster(self,
                        atoms: Atoms, 
                        cluster_indices: list, 
                        new_element: str,
                        new_atom_count: int,
                        inplace=False) -> Atoms:
        """
        Replace a cluster of atoms with a new element and optimize the positions.
        """

        # Make a copy of the input atoms object
        new_atoms = atoms.copy()
        
        # Obtain the initial cluster information
        centroid = self.pbc_aware_centroid(
            atoms.positions[cluster_indices], atoms.cell, atoms.pbc)
        
        # Generate new atom positions (Fibonacci sphere distribution)
        sphere = self.fibonacci_sphere(new_atom_count, radius=1.0)
        element_radius = covalent_radii[atomic_numbers[new_element]]
        neighbor_radius = np.max([covalent_radii[atom.number] for atom in atoms])
        safe_radius = 2 * (element_radius + neighbor_radius)  # 2 * sum of radii
        
        # Scale initial positions using safe radius
        initial_positions = centroid + sphere * safe_radius * new_atom_count**0.33

        # Define the optimization procedure
        def loss_function(positions):
            positions = positions.reshape(-1, 3)
            trial = new_atoms.copy()
            indices = self.replace_atoms(
                trial, cluster_indices, new_element, positions)
            return self.calculate_loss(trial, indices, centroid)
        
        result = minimize(loss_function, initial_positions.flatten(),
                          method='L-BFGS-B', 
                          options={'maxiter': self.optimization_steps})
        
        # Update the atom positions
        optimized_positions = result.x.reshape(-1, 3)

        if inplace:
            self.replace_atoms(atoms, cluster_indices,
                               new_element, optimized_positions)
            return atoms
        else:
            self.replace_atoms(new_atoms, cluster_indices, 
                           new_element, optimized_positions)
        
            return new_atoms
    
    def calculate_loss(self, atoms: Atoms, indices: np.ndarray, centroid: np.ndarray):
        """
        Calculate the loss function for the optimization.
        The loss function is a weighted sum of three terms: repulsion, self-interaction, 
        and attraction.
        """

        # Initialize cutoffs. Only the selected atoms have a cutoff.
        cutoffs = np.zeros(len(atoms))
        cutoffs[indices] = self.cutoff

        # Build neighbor lists
        nl = NeighborList(cutoffs=cutoffs, skin=0.3, self_interaction=False, 
                          bothways=True)
        nl.update(atoms)

        # Repulsion term: the new positions should be not too close to their neighbors.
        repulsion = 0.0
        R = atoms.positions
        for idx in indices:
            neighbors, offsets = nl.get_neighbors(idx)
            D = R[neighbors] + offsets @ atoms.get_cell() - R[idx]
            repulsion += 1.0 / np.min(np.linalg.norm(D, axis=1))
        
        # Self-interaction term: the interatomic distances should be uniform.
        self_dists = get_distances(
            atoms.positions[indices], 
            atoms.positions[indices], 
            cell=atoms.cell, pbc=True)[1]
        dists = []
        for i in range(len(indices)):
            dists.extend(self_dists[i, i+1:])
        dists = np.array(dists)
        self_interaction = np.linalg.norm(dists - dists.mean())
                
        # Attraction term: the new positions should be not far from the centroid
        new_positions = atoms.positions[indices]
        center_disp = get_distances(new_positions, centroid, cell=atoms.cell, pbc=True)
        center_term = center_disp[1].mean()
        
        # Weighted sum (adjusted weights)
        return (
            0.6 * repulsion +
            0.2 * self_interaction +
            0.2 * center_term
        )

    def fibonacci_sphere(self, n, radius=1.0, min_distance=1.0):
        """
        Generate n points on a sphere using the Fibonacci lattice method.
        """
        points = []
        while len(points) < n:
            indices = np.arange(len(points), len(points)+1000)
            phi = np.arccos(1 - 2*(indices+0.5)/(n+1000))
            theta = np.pi * (1 + 5**0.5) * indices
            
            new_points = np.stack([
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi)
            ], axis=1)
            
            # Filter points that are too close
            if points:
                distances = np.linalg.norm(new_points[:, None] - np.array(points), 
                                           axis=2)
                mask = np.all(distances > min_distance, axis=1)
                new_points = new_points[mask]
                
            points.extend(new_points[:n-len(points)])
        
        return np.array(points[:n])


def find_cluster_neighbor_list_method(atoms: Atoms, center_atom_idx: int, 
                                      cluster_size: int, cutoff: float = 5.0) -> list:
    """
    Find approximate minimum-distance cluster using neighbor lists with element 
    filtering.
    
    Args:
        atoms: ASE Atoms object
        M: Cluster size (must include center_atom_idx)
        center_atom_idx: Index of the mandatory atom
        eltype: Target element symbol
        cutoff: Neighbor search radius (angstrom)
        
    Returns:
        List of atom indices in the cluster
    
    """
    # Validate inputs
    assert 1 <= cluster_size < 5, "cluster_size must be between 2 and 4"
    
    # Step 1: Filter target element atoms
    eltype = atoms[center_atom_idx].symbol
    target_indices = [i for i, atom in enumerate(atoms) if atom.symbol == eltype]
    assert len(target_indices) >= cluster_size, "Insufficient target element atoms"
    
    # Step 2: Build neighbor list (only for target elements)
    # Create a mask to ignore non-target atoms
    nl = NeighborList(
        cutoffs=[cutoff if i in target_indices else 0.0 for i in range(len(atoms))],
        self_interaction=False,
        bothways=True
    )
    nl.update(atoms)
    
    # Step 3: Get neighbors of center_atom_idx (already filtered by target_indices)
    neighbors, offsets = nl.get_neighbors(center_atom_idx)
    
    # Step 4: Select closest M-1 neighbors
    valid_neighbors = []
    for n, offset in zip(neighbors, offsets):
        if atoms[n].symbol != eltype:
            continue
        true_position = atoms.positions[n] + np.dot(offset, atoms.get_cell())
        distance = np.linalg.norm(true_position - atoms.positions[center_atom_idx])
        valid_neighbors.append((distance, n))
    
    # Pick top M-1 closest valid neighbors
    valid_neighbors.sort()
    return [center_atom_idx] + [n for _, n in valid_neighbors[:cluster_size - 1]]
