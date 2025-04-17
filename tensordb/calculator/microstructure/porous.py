# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict
from tensordb.calculator.calculator import VaspCalculator
from tensordb.sampler import BaseSampler
from tensordb.utils import getitem


class VaspPorousCalculator(VaspCalculator):
    """
    This sampler is used to create porous structures by randomly removing atoms from 
    AIMD snapshots.
    """

    def __init__(self, root, config):
        super().__init__(root, config)
        self.workdir = self.root / "porous"
        
        params = getitem(self.config, ["porosity", ])
        if "porosity" not in params:
            raise ValueError(
                "The 'porosity' key is not found in the 'porosity' section.")
        porosity = params["porosity"]
        if isinstance(porosity, (int, float)):
            self.get_porosity = lambda : porosity
        else:
            assert len(porosity) == 2, "The 'porosity' should be a list of two floats."
            pmin, pmax = porosity
            if pmin > pmax:
                pmin, pmax = pmax, pmin
            if pmin <= 1:
                print("Warning: The minimum porosity should be larger than 1!")
                pmin = 1.01
            if pmax > 3:
                print(f"Warning: The maximum porosity is {pmax}, you should know what "
                      f"you are doing!")
            self.get_porosity = lambda : np.random.uniform(pmin, pmax)
        self.sampling_interval = params.get("interval", 500)
    
    @property
    def random_seed(self):
        return 0

    def task_iterator(self):
        """
        Iterate through all high-precision DFT calculation job dirs.
        """
        return self.workdir.glob("*atoms/group*/task*")

    def may_modify_atoms(self, atoms):
        """
        Create porous structures by randomly removing atoms.
        """
        porosity = self.get_porosity()
        n = len(atoms)
        nd = max(1, n - int(n / porosity))
        indices = np.random.choice(n, nd, replace=False)
        obj = atoms.copy()
        del obj[indices]
        return obj

    def create_tasks(self, samplers: Dict[str, BaseSampler], **kwargs):
        """
        Create VASP high precision DFT calculation tasks.
        For VaspPorousCalculator, the random seed is fixed to 0.
        """
        np.random.seed(self.random_seed)
        super().create_tasks(samplers, **kwargs)
