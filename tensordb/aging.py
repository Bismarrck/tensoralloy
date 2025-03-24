# -*- coding: utf-8 -*-
# This module defines the calculator for creating irradiation-induced aging structures.

import numpy as np
from dataclasses import dataclass
from typing import List
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from tensordb.calculator import VaspCalculator
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
class HeliumBubbleInjection(Transmutation):
    """
    The dataclass for defining an irradiation-induced helium bubble injection.
    """
    ratio = 1
    dist_mean = 1.5
    dist_std = 0.5


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
        self.interval = params.get("interval", self.defaults["interval"])
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
        self.helium_bubbles = []
        for src, value in params.get("helium_bubble", {}).items():
            if not isinstance(value, dict):
                raise ValueError(f"The value of {key} should be a dictionary.")
            prob = value.get("prob", self.defaults["prob"])
            ratio = value.get("ratio", self.defaults["ratio"])
            nmax = value.get("nmax", self.defaults["nmax"])
            dist_mean = value.get("dist_mean", self.defaults["dist_mean"])
            dist_std = value.get("dist_std", self.defaults["dist_std"])
            self.helium_bubbles.append(
                HeliumBubbleInjection(
                    src=src, dst="He", prob=prob, ratio=ratio,
                    nmax=nmax, dist_mean=dist_mean, dist_std=dist_std))

    @property
    def random_seed(self):
        return 1

    def may_inject_helium_bubble(self, atoms: Atoms, indices: np.ndarray):
        """
        The method for injecting a small helium cluster into the structure.
        """
        n = len(atoms)
        # Reserve the space for the points
        points = np.zeros((n * 2, 3))
        for i in indices:
            if atoms.info["modified"][i]:
                continue
            points[0] = atoms[i].position
            D = neighbor_list('D', atoms, cutoff=self.cutoff)
            points[1: len(D) + 1] = D + points[0]

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
        
        # 3. Loop through each transmutation.
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
        
        # 4. Loop through each helium bubble injection.
        for helium_bubble in self.helium_bubbles:
            self.may_inject_helium_bubble(obj, shuffled_indices)
        
        return obj

    def create_tasks(self, samplers: List[BaseSampler], **kwargs):
        """
        Create VASP high precision DFT calculation tasks.
        """
        super().create_tasks(samplers, **kwargs)


def replace_point_with_gaussian(points: np.ndarray, k=2, d1=1.0, d2=0.1, 
                                d3=2.0, lambda_param=1.0, max_iter=100):
    """
    Randomly replace a point in a 3D point set with k new points.
    * The distance between the new points and the replaced point is less than d3.
    * The distance between the new points follows a Gaussian distribution N(d1, d2).
    * The new points are as far away from other points as possible.

    Parameters:
    * points: The original point set (a list of 3D coordinates).
    * k: The number of new points to generate (2 to N).
    * d1: The mean of the Gaussian distribution.
    * d2: The standard deviation of the Gaussian distribution.
    * d3: The maximum distance to the replaced point.
    * lambda_param: The weight of the Gaussian distribution constraint.
    * max_iter: The maximum number of optimization iterations.

    """
    
    # 1. Setup
    replace_idx = 0
    P = points[replace_idx]
    other_points = points[1:]
    
    # 2. Generate the initial points (randomly distributed around P)
    np.random.seed(3)
    initial_guess = []
    for _ in range(k):
        # Generate a random direction and normalize it
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        # Generate a random radius (0 to d3)
        radius = np.random.uniform(0, d3)
        Q = P + direction * radius
        initial_guess.extend(Q)
    initial_guess = np.array(initial_guess)
    
    # 3. The objective function. 
    # The rule is maximizing the minimum distance with Gaussian distribution constraint.
    def objective(vars, other_points, P, d1, d2, lambda_param):
        new_points = vars.reshape(-1, 3)
        k = new_points.shape[0]
        
        # Calculate the minimum distance to other points
        if len(other_points) > 0:
            min_dists = cdist(new_points, other_points)
            min_d = np.min(min_dists)
        else:
            min_d = np.inf
        
        # Calculate the mean and variance of the pairwise distances
        pairwise_dists = cdist(new_points, new_points)
        triu_indices = np.triu_indices(k, 1)
        valid_dists = pairwise_dists[triu_indices]
        
        if len(valid_dists) == 0:
            mean_dist, var_dist = 0.0, 0.0
        else:
            mean_dist = np.mean(valid_dists)
            var_dist = np.var(valid_dists)
        
        # Maximize the minimum distance + Gaussian distribution penalty
        loss = -min_d + lambda_param * ((mean_dist - d1)**2 + (var_dist - d2**2)**2)
        return loss
    
    # 4. Define constraints (the distance between each point and P <= d3)
    constraints = []
    for i in range(k):
        def make_con(i):
            def con_func(vars):
                new_points = vars.reshape(-1, 3)
                return d3 - np.linalg.norm(new_points[i] - P)
            return con_func
        constraints.append({'type': 'ineq', 'fun': make_con(i)})
    
    # 5. Do optimization
    result = minimize(
        objective, initial_guess,
        args=(other_points, P, d1, d2, lambda_param),
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': max_iter}
    )
    
    # 6. Process the result
    if result.success:
        new_points = result.x.reshape(-1, 3)
    else:
        new_points = initial_guess.reshape(-1, 3)
    
    return [tuple(p) for p in other_points] + [tuple(p) for p in new_points]
