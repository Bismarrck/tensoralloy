#!coding-utf-8
"""
Equation of State for computing equilibrium energy, volume and bulk modulus.
"""
from __future__ import print_function, absolute_import

import numpy as np
import warnings
import ase.eos

from scipy.optimize import curve_fit
from ase.eos import parabola, p3, birchmurnaghan, birch, murnaghan, antonschmidt
from ase.eos import vinet, pouriertarantola
from ase.units import GPa

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def rose(V, E0, B0, BP, V0, beta=0.005):
    """
    The Rose Universal Equation of State
    """
    alpha = np.sqrt(-9 * V0 * B0 / E0)
    x = (V / V0)**(1. / 3.) - 1
    ax = alpha * x
    return E0 * (1 + ax + beta * ax**3 * (2 * x + 3) / (x + 1)**2) * np.exp(-ax)


class EquationOfState(ase.eos.EquationOfState):
    """
    A modified implementation of EquationOfState.
    """

    def __init__(self, volumes, energies, eos='sj', beta=0.005):
        """
        Initialization method.
        """
        super(EquationOfState, self).__init__(volumes, energies, eos)

        self.func = None
        self.beta = beta

    def fit(self, warn=False):
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

        if self.eos_string == 'sj':
            return self.fit_sjeos()
        elif self.eos_string == "pouriertarantola":
            self.func = pouriertarantola
        elif self.eos_string == "birch":
            self.func = birch
        elif self.eos_string == "birchmurnaghan":
            self.func = birchmurnaghan
        elif self.eos_string == "murnaghan":
            self.func = murnaghan
        elif self.eos_string == "rose":
            def rose_wrap(*args):
                return rose(*args, beta=self.beta)
            self.func = rose_wrap
        elif self.eos_string == "vinet":
            self.func = vinet
        elif self.eos_string == "antonschmidt":
            self.func = antonschmidt
        else:
            raise ValueError("")

        p0 = [min(self.e), 1, 1]
        popt, pcov = curve_fit(parabola, self.v, self.e, p0)

        parabola_parameters = popt
        # Here I just make sure the minimum is bracketed by the volumes
        # this if for the solver
        minvol = min(self.v)
        maxvol = max(self.v)

        # the minimum of the parabola is at dE/dV = 0, or 2 * c V +b =0
        c = parabola_parameters[2]
        b = parabola_parameters[1]
        a = parabola_parameters[0]
        parabola_vmin = -b / 2 / c

        # evaluate the parabola at the minimum to estimate the groundstate
        # energy
        E0 = parabola(parabola_vmin, a, b, c)
        # estimate the bulk modulus from Vo * E''.  E'' = 2 * c
        B0 = 2 * c * parabola_vmin

        if self.eos_string == 'antonschmidt':
            BP = -2
        else:
            BP = 4

        initial_guess = [E0, B0, BP, parabola_vmin]

        # now fit the equation of state
        p0 = initial_guess
        popt, pcov = curve_fit(self.func, self.v, self.e, p0)

        self.eos_parameters = popt

        if self.eos_string == 'p3':
            c0, c1, c2, c3 = self.eos_parameters
            # find minimum E in E = c0 + c1 * V + c2 * V**2 + c3 * V**3
            # dE/dV = c1+ 2 * c2 * V + 3 * c3 * V**2 = 0
            # solve by quadratic formula with the positive root

            a = 3 * c3
            b = 2 * c2
            c = c1

            self.v0 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            self.e0 = p3(self.v0, c0, c1, c2, c3)
            self.B = (2 * c2 + 6 * c3 * self.v0) * self.v0
        else:
            self.v0 = self.eos_parameters[3]
            self.e0 = self.eos_parameters[0]
            self.B = self.eos_parameters[1]

        if warn and not (minvol < self.v0 < maxvol):
            warnings.warn(
                'The minimum volume of your fit is not in '
                'your volumes.  You may not have a minimum in your dataset!')

        residual = np.linalg.norm(
            self.e - self.func(self.v, *self.eos_parameters))

        return self.v0, self.e0, self.B / GPa, residual
