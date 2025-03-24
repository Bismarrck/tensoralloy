# -*- coding: utf-8 -*-
# This module defines the calculator for creating irradiation-induced aging structures.

import numpy as np
from dataclasses import dataclass
from typing import List
from ase import Atoms
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
    [aging.transmutation]
    U-Th = {"prob": 0.5, "nmax": 1}
    U-He = {"prob": 0.5, "ratio": 2, "nmax": 1, "dist_mean": 1.5, "dist_std": 0.5}

    Explanations: 
    1. Every 1/100 snapshots are used to create aging structures.
    2. Element U to Th
    * probability is 50%
    * at most 1 U atom is transmutated.
    
    3. Element U to He: 
    * probability is 50%
    * the ratio of U to He is 1:1 or 2:1
    * at most 1 U atom is transmutated
    * the distance between He and He is 1.5 A with a standard deviation of 0.5 A

    """
    defaults = {"prob": 0.5, "ratio": 1, "nmax": 1, "dist_mean": 1.5, "dist_std": 0.5}
    
    def __init__(self, root, config):
        super().__init__(root, config)
        self.workdir = self.root / "aging"

        params = getitem(self.config, ["aging", ])
        self.interval = params.get("interval", 100)
        transmutations = params.get("transmutation", {})
        if len(transmutations) == 0:
            raise ValueError("The 'transmutation' section is empty!")
        self.transmutations = []
        for key, value in transmutations.items():
            if not isinstance(value, dict):
                raise ValueError(f"The value of {key} should be a dictionary.")
            src = key.split("-")[0]
            if src not in self.species:
                raise ValueError(
                    f"The source element {src} is not in the species list.")
            dst = key.split("-")[1]
            prob = value.get("prob", self.defaults["prob"])
            ratio = value.get("ratio", self.defaults["ratio"])
            nmax = value.get("nmax", self.defaults["nmax"])
            if dst != "He" and ratio > 1:
                raise ValueError(
                    f"The ratio of {src} to {dst} should be less than 1.")
            self.transmutations.append(
                Transmutation(src, dst, prob, ratio, nmax, 0))
    
    @property
    def random_seed(self):
        return 1

    def may_inject_helium_bubble(self, atoms: Atoms, indices: np.ndarray):
        """
        The method for injecting a small helium cluster into the structure.
        """
        pass

    def may_modify_atoms(self, atoms: Atoms):
        """
        Create aging structures.
        """
        # Reset the counter
        for transmutation in self.transmutations:
            transmutation.used = 0
        # 1. Copy the input `Atoms` object.
        obj = atoms.copy()
        n = len(obj)
        # 2. Shuffle the atoms to modify.
        obj.info["modified"] = np.zeros(n, dtype=bool)
        shuffled_indices = np.random.permutation(n)
        # 3. Loop through each transmutation.
        for transmutation in self.transmutations:
            # For helium bubble injection, a separate method is used as it is complex.
            if transmutation.dst == "He":
                self.may_inject_helium_bubble(obj, shuffled_indices)
            else:
                # For normal transmutation, we just replace `src` with `dst`.
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
        for i in range(n):
            print(f"{i:2d} {atoms[i].symbol:2s} -> {obj[i].symbol:2s}")
        return obj

    def create_tasks(self, samplers: List[BaseSampler], **kwargs):
        """
        Create VASP high precision DFT calculation tasks.
        """
        super().create_tasks(samplers, **kwargs)
