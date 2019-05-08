#!coding-utf-8
"""
Equation of State for computing equilibrium energy, volume and bulk modulus.
"""
from __future__ import print_function, absolute_import

import numpy as np
import ase.eos

from ase.units import GPa

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EquationOfState(ase.eos.EquationOfState):
    """
    A modified implementation of EquationOfState.
    """

    def __init__(self, volumes, energies, eos='sj'):
        """
        Initialization method.
        """
        super(EquationOfState, self).__init__(volumes, energies, eos)

    def fit(self):
        """
        Calculate volume, energy, and bulk modulus.

        Returns
        -------
        v0 : float
            The equilibrium volume.
        e0 : float
            The equilibrium energy.
        B : float
            The bulk modulus, in GPa.
        residual : float
            The norm of the residual vector of the fit. The unit is eV. A good
            fit typically means the residual is smaller than 0.005 eV.

        """
        # Fit the EOS
        v0, e0, B = super(EquationOfState, self).fit()
        residual = np.linalg.norm(
            self.e - self.func(self.v, *self.eos_parameters))

        return v0, e0, B / GPa, residual
