#!coding=utf-8
"""
This module defines a demo of TensorAlloy's phonon spectrum function.
"""
from __future__ import print_function, absolute_import

import numpy as np

from tensoralloy import TensorAlloyCalculator

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def phonon_spectrum_example():
    """
    A demo of plotting phonon spectrum of Ni/fcc.
    """
    from ase.build import bulk
    from os.path import join
    from tensoralloy.test_utils import test_dir

    band_paths = [
        [np.array([0, 0, 0]),
         np.array([1 / 2, 0, 1 / 2]),
         np.array([1 / 2, 1 / 4, 3 / 4]),
         np.array([1 / 2, 1 / 2, 1]),
         np.array([3 / 8, 3 / 8, 3 / 4]),
         np.array([0, 0, 0]),
         np.array([1 / 2, 1 / 2, 1 / 2])]
    ]

    labels = [
        r"$\Gamma$",
        r"$\mathrm{X}$",
        r"$\mathrm{W}$",
        r"$\mathrm{X}$",
        r"$\mathrm{K}$",
        r"$\Gamma$",
        r"$\mathrm{L}$",
    ]

    atoms = bulk('Ni')
    calc = TensorAlloyCalculator(join(test_dir(), 'models', 'Ni.sf.r40.pb'))
    atoms.calc = calc
    calc.get_phonon_spectrum(
        atoms,
        primitive_axes='auto',
        supercell=(4, 4, 4),
        numeric_hessian=False,
        band_paths=band_paths,
        band_labels=labels,
        npoints=101,
        use_wavenumber=True, plot_vertical_lines=True,
        verbose=True)


if __name__ == "__main__":
    phonon_spectrum_example()
