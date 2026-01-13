# -*- coding: utf-8 -*-
# This module defines the calculator for creating irradiation-induced aging structures.

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
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
    algorithm: str = "fibonacci"

    def __post_init__(self):
        assert 1 <= self.max_target_size <= 5
        assert 1 <= self.max_bubble_size <= 10
        assert self.max_ratio <= 4.0
        assert 0.5 <= self.min_ratio <= self.max_ratio


class VaspAgingCalculator(VaspCalculator):
    """
    The VASP calculator for creating irradiation-induced aging structures.
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
            # Note: We don't check if src is in self.species here strictly, 
            # as the species list might update or depend on the specific loaded atoms.
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
                max_ratio=args.get("max_ratio", 3.0),
                min_ratio=args.get("min_ratio", 1.0),
                cutoff=args.get("cutoff", self.defaults["cutoff"]),
                algorithm=args.get("algorithm", "fibonacci")
            )

    @property
    def random_seed(self):
        return 1

    def _get_valid_bubble_sizes(self, args: HeliumBubbleInjection) -> List[Tuple[int, int]]:
        """Helper to find valid (cluster_size, bubble_size) pairs."""
        valid_pairs = []
        for c_size in range(1, args.max_target_size + 1):
            for b_size in range(1, args.max_bubble_size + 1):
                ratio = b_size / c_size
                if args.min_ratio <= ratio <= args.max_ratio:
                    valid_pairs.append((c_size, b_size))
        return valid_pairs

    def may_inject_helium_bubble(self, atoms: Atoms, center_idx: int):
        """
        The method for injecting a small helium bubble into the structure.
        Modifies atoms in-place.
        """
        if self.helium_bubble_injection is None:
            return

        # Initialize the helium bubble injector
        args = self.helium_bubble_injection
        algo = self.config.get("helium_bubble_algorithm", "fibonacci").lower()
        if algo == "dense_packing":
            injector = DensePackingHeliumBubbleInjector(cutoff=args.cutoff)
        else:
            injector = FibnonacciSphereHeliumBubbleInjector(cutoff=args.cutoff)
        
        # 1. Determine sizes deterministically to avoid infinite loops
        valid_pairs = self._get_valid_bubble_sizes(args)
        
        if not valid_pairs:
            # Fallback or silent return if configuration allows no valid bubbles
            return 

        # Pick a random valid pair
        idx = np.random.randint(0, len(valid_pairs))
        cluster_size, bubble_size = valid_pairs[idx]

        # Do the injection
        # Note: This changes the length of 'atoms', invalidating subsequent indices in the caller
        injector.inject(atoms, center_idx, cluster_size, bubble_size, inplace=True)

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
        # Use a local boolean mask instead of obj.info to track status.
        # obj.info can become desynchronized if atoms are deleted later.
        modified_mask = np.zeros(n, dtype=bool)
        shuffled_indices = np.random.permutation(n)
        
        # 3. Apply Transmutations FIRST.
        # Transmutations only change symbols, they do not change the number of atoms.
        # Therefore, indices remain valid.
        for transmutation in self.transmutations:
            for i in shuffled_indices:
                if modified_mask[i]:
                    continue
                
                if obj[i].symbol == transmutation.src:
                    # Check the probability
                    if np.random.uniform(0, 1) <= transmutation.prob:
                        obj[i].symbol = transmutation.dst
                        modified_mask[i] = True
                        transmutation.used += 1
                        if transmutation.used >= transmutation.nmax:
                            break
        
        # 4. Apply Helium Bubble Injection LAST.
        # This removes atoms and changes the index mapping.
        # We select a center from atoms that have NOT been modified yet.
        if self.helium_bubble_injection is not None:
            for i in shuffled_indices:
                if modified_mask[i]:
                    continue
                
                # Check if this atom is a valid target for the bubble
                if obj[i].symbol == self.helium_bubble_injection.target:
                    # Apply injection and break (usually only one bubble per snapshot is intended 
                    # based on logic, or logic can be extended)
                    self.may_inject_helium_bubble(obj, i)
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
        if len(positions) == 0:
            return np.array([0.0, 0.0, 0.0])
            
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
        """
        # Find neighbors to remove
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
    
    def replace_atoms(self, atoms: Atoms, old_indices: List[int], 
                      new_element: str, new_positions: np.ndarray) -> List[int]:
        """
        Replace a group of atoms with new atoms in the structure.
        Returns the indices of the newly added atoms.
        """
        # Delete old atoms. ASE handles list deletion safely.
        # Note: After deletion, indices of subsequent atoms shift.
        del atoms[old_indices]
        
        new_indices = []
        # Append new atoms
        start_idx = len(atoms)
        for i, pos in enumerate(new_positions):
            atoms.append(Atom(new_element, position=pos))
            new_indices.append(start_idx + i)
            
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
        # If inplace, we work on the passed object, otherwise copy it
        work_atoms = atoms if inplace else atoms.copy()
        
        # Obtain the initial cluster information
        # Note: We must get positions BEFORE deleting atoms
        centroid = self.pbc_aware_centroid(
            work_atoms.positions[cluster_indices], work_atoms.cell, work_atoms.pbc)
        
        # Determine scaling for initial placement
        element_radius = covalent_radii[atomic_numbers[new_element]]
        
        # Safely get neighbor radius (handle case where list might be empty if we removed everything, 
        # though unlikely in bulk)
        existing_radii = [covalent_radii[atom.number] for atom in work_atoms]
        neighbor_radius = np.max(existing_radii) if existing_radii else 1.5
        
        safe_radius = 2 * (element_radius + neighbor_radius)
        
        # Generate new atom positions (Fibonacci sphere distribution)
        sphere = self.fibonacci_sphere(new_atom_count, radius=1.0)
        initial_positions = centroid + sphere * safe_radius * (new_atom_count**0.33)

        # We need a temporary object for optimization to not mess up the main object 
        # until the end (or to handle the loss function calculation properly)
        
        # Strategy: Perform replacement on a copy to run the optimizer
        opt_atoms = work_atoms.copy()
        
        # Replace atoms in the copy to get the NEW indices
        added_indices = self.replace_atoms(
            opt_atoms, cluster_indices, new_element, initial_positions)
        
        # Define the optimization procedure
        def loss_function(flat_positions):
            current_positions = flat_positions.reshape(-1, 3)
            # Update positions of the specific helium atoms
            opt_atoms.positions[added_indices] = current_positions
            return self.calculate_loss(opt_atoms, added_indices, centroid)
        
        # Optimize
        result = minimize(loss_function, initial_positions.flatten(),
                          method='L-BFGS-B', 
                          options={'maxiter': self.optimization_steps})
        
        optimized_positions = result.x.reshape(-1, 3)

        # Apply final result to the actual target atoms
        # If we were already working inplace on 'atoms' (work_atoms == atoms), 
        # we need to perform the replacement now using the optimized positions.
        # Note: We haven't touched 'work_atoms' yet, only 'opt_atoms'.
        self.replace_atoms(work_atoms, cluster_indices, new_element, optimized_positions)
        
        return work_atoms
    
    def calculate_loss(self, atoms: Atoms, indices: List[int], centroid: np.ndarray):
        """
        Calculate the loss function for the optimization.
        """
        # Ensure indices is a numpy array for indexing
        indices_arr = np.array(indices)

        # Initialize cutoffs. Only the selected atoms have a cutoff.
        cutoffs = np.zeros(len(atoms))
        cutoffs[indices_arr] = self.cutoff

        # Build neighbor lists
        nl = NeighborList(cutoffs=cutoffs, skin=0.3, self_interaction=False, 
                          bothways=True)
        nl.update(atoms)

        # Repulsion term
        repulsion = 0.0
        R = atoms.positions
        
        # We only care about neighbors of the inserted atoms
        for idx in indices:
            neighbors, offsets = nl.get_neighbors(idx)
            if len(neighbors) == 0:
                continue
            D = R[neighbors] + offsets @ atoms.get_cell() - R[idx]
            dists = np.linalg.norm(D, axis=1)
            # Avoid division by zero
            dists = np.maximum(dists, 1e-2)
            repulsion += np.sum(1.0 / dists)
        
        # Self-interaction term: equidistribution among He atoms
        if len(indices) > 1:
            # get_distances returns a tuple, index 1 is the distance matrix
            dist_mat = get_distances(
                atoms.positions[indices_arr], 
                cell=atoms.cell, pbc=atoms.pbc)[1]
            
            # Extract upper triangle to avoid duplicates and self-distances
            dists = dist_mat[np.triu_indices(len(indices), k=1)]
            
            if len(dists) > 0:
                self_interaction = np.std(dists) # Minimize standard deviation
            else:
                self_interaction = 0.0
        else:
            self_interaction = 0.0
                
        # Attraction term: keep close to centroid
        new_positions = atoms.positions[indices_arr]
        # Calculate distance to centroid (single point)
        # Note: simple norm is sufficient if we assume optimization starts close to centroid
        # but technically should respect PBC.
        disp_to_center = new_positions - centroid
        # Apply MIC to centroid distance
        disp_to_center, _ = find_mic(disp_to_center, atoms.cell, atoms.pbc)
        center_dists = np.linalg.norm(disp_to_center, axis=1)
        center_term = np.mean(center_dists)
        
        return (
            0.6 * repulsion +
            0.2 * self_interaction +
            0.2 * center_term
        )

    def fibonacci_sphere(self, n, radius=1.0, min_distance=0.1):
        """
        Generate n points on a sphere using the Fibonacci lattice method.
        """
        points = []
        # Standard Fibonacci sphere algorithm
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2 if n > 1 else 0  # y goes from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y

            points.append([x * radius, y * radius, z * radius])

        return np.array(points)


def find_cluster_neighbor_list_method(atoms: Atoms, center_atom_idx: int, 
                                      cluster_size: int, cutoff: float = 5.0) -> list:
    """
    Find approximate minimum-distance cluster using neighbor lists with element 
    filtering.
    """
    # Validate inputs
    # Fixed assertion: allow cluster_size up to 5 inclusive
    assert 1 <= cluster_size <= 5, "cluster_size must be between 1 and 5"
    
    if cluster_size == 1:
        return [center_atom_idx]

    # Step 1: Filter target element atoms
    eltype = atoms[center_atom_idx].symbol
    target_indices = [i for i, atom in enumerate(atoms) if atom.symbol == eltype]
    
    if len(target_indices) < cluster_size:
        raise ValueError(f"Insufficient target element atoms of type {eltype}")
    
    # Step 2: Build neighbor list
    # Use a mask for cutoffs to speed up: only search around target atoms
    cutoffs = [cutoff if i in target_indices else 0.0 for i in range(len(atoms))]
    nl = NeighborList(
        cutoffs=cutoffs,
        self_interaction=False,
        bothways=True
    )
    nl.update(atoms)
    
    # Step 3: Get neighbors of center_atom_idx
    neighbors, offsets = nl.get_neighbors(center_atom_idx)
    
    # Step 4: Select closest M-1 neighbors that match the element type
    valid_neighbors = []
    for n, offset in zip(neighbors, offsets):
        if atoms[n].symbol != eltype:
            continue
        true_position = atoms.positions[n] + np.dot(offset, atoms.get_cell())
        distance = np.linalg.norm(true_position - atoms.positions[center_atom_idx])
        valid_neighbors.append((distance, n))
    
    # Sort by distance
    valid_neighbors.sort()
    
    # Pick top M-1 closest valid neighbors
    # Check if we found enough neighbors
    needed = cluster_size - 1
    if len(valid_neighbors) < needed:
        # Fallback: if cutoff was too small, just pick the center to avoid crash, 
        # or raise error. Here we raise error.
        raise ValueError("Cutoff too small to find required cluster size.")

    return [center_atom_idx] + [n for _, n in valid_neighbors[:needed]]


from scipy.spatial.transform import Rotation

class DensePackingHeliumBubbleInjector(HeliumBubbleInjector):
    """
    Injects Helium atoms by cutting a cluster from a dense FCC lattice 
    and optimizing the geometry. This is better for high-density bubbles 
    (ratio > 1) compared to the Fibonacci shell method.
    """

    def replace_cluster(self,
                        atoms: Atoms, 
                        cluster_indices: list, 
                        new_element: str,
                        new_atom_count: int,
                        inplace=False) -> Atoms:
        
        work_atoms = atoms if inplace else atoms.copy()
        
        # 1. Calculate Centroid
        centroid = self.pbc_aware_centroid(
            work_atoms.positions[cluster_indices], work_atoms.cell, work_atoms.pbc)
        
        # 2. Determine Scale
        # He-He distance in high pressure bubbles is typically 2.0 - 2.5 Angstroms.
        # We can estimate it based on the available volume of the removed cluster.
        # Volume of removed host atoms ~ N_host * (4/3 * pi * r_host^3)
        # Volume of He atoms ~ N_he * (4/3 * pi * r_he^3)
        # We start with a dense guess (2.2 A spacing) and let the optimizer relax it.
        initial_spacing = 2.2 

        # 3. Generate FCC Cluster (Dense Packing)
        # Generate points centered at (0,0,0)
        points = self.generate_fcc_cluster(new_atom_count, spacing=initial_spacing)
        
        # 4. Random Rotation
        # Rotate the bubble so the He lattice isn't perfectly aligned with the host lattice
        # which prevents artificial channeling effects during relaxation.
        rot = Rotation.random()
        points = rot.apply(points)

        # 5. Move to Centroid
        initial_positions = points + centroid

        # 6. Optimization (Same as before)
        opt_atoms = work_atoms.copy()
        added_indices = self.replace_atoms(
            opt_atoms, cluster_indices, new_element, initial_positions)
        
        def loss_function(flat_positions):
            current_positions = flat_positions.reshape(-1, 3)
            opt_atoms.positions[added_indices] = current_positions
            return self.calculate_loss(opt_atoms, added_indices, centroid)
        
        result = minimize(loss_function, initial_positions.flatten(),
                          method='L-BFGS-B', 
                          options={'maxiter': self.optimization_steps})
        
        optimized_positions = result.x.reshape(-1, 3)
        
        # 7. Apply to final object
        self.replace_atoms(work_atoms, cluster_indices, new_element, optimized_positions)
        
        return work_atoms

    def generate_fcc_cluster(self, n, spacing=2.2):
        """
        Generates the N closest points to the origin from a generated FCC grid.
        """
        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])

        # Create a grid large enough to hold N atoms
        # An FCC unit cell has 4 atoms. We need ceil(N/4) cells roughly.
        # Make a grid of roughly cubic size.
        dim = int(np.ceil((n / 4)**(1/3))) + 1
        
        basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]) * spacing * np.sqrt(2)
        
        points = []
        for x in range(-dim, dim+1):
            for y in range(-dim, dim+1):
                for z in range(-dim, dim+1):
                    offset = np.array([x, y, z]) * spacing * np.sqrt(2)
                    points.append(basis + offset)
        
        points = np.vstack(points)
        
        # Sort by distance from origin
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        
        # Return the N closest points to origin
        # Use simple centering to ensure the center of mass is close to 0
        selected = points[sorted_indices[:n]]
        center_of_mass = np.mean(selected, axis=0)
        return selected - center_of_mass
