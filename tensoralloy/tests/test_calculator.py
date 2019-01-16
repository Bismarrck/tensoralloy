# coding=utf-8
"""
This module defines unit tests of `TensorAlloyCalculator`.
"""
from __future__ import print_function, absolute_import

import numpy as np
import nose

from nose.tools import assert_almost_equal, assert_raises
from os.path import join
from ase.db import connect
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.utils import Defaults
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_calculator_with_nickel():
    """
    Test `TensorAlloyCalculator` using the Ni database.
    """
    graph_path = join(test_dir(), 'checkpoints', 'Ni-k2', 'Ni.pb')
    sqlite3_file = join(test_dir(), 'checkpoints', 'Ni-k2', 'Ni.db')

    def run_test():
        """ A wrapper. """
        calc = TensorAlloyCalculator(graph_path)

        db = connect(sqlite3_file)
        _, tests = train_test_split(np.arange(1, 1 + len(db)),
                                    random_state=Defaults.seed, test_size=100)

        y_true = []
        y_pred = []
        for index in tests:
            atoms = db.get_atoms(f'id={index}')
            y_true.append(atoms.get_total_energy())
            y_pred.append(calc.get_potential_energy(atoms))

        y_mae = mean_absolute_error(y_true, y_pred)
        assert_almost_equal(y_mae, 0.00769633, delta=1e-7)

        calc.get_forces(db.get_atoms('id=1'))

    # The model file should be updated because the arg `k_max` is replaced with
    # `angular`.
    with assert_raises(Exception):
        run_test()


if __name__ == "__main__":
    nose.run()
