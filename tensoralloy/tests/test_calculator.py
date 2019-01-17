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
from tensoralloy.test_utils import test_dir, datasets_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_calculator_with_qm7():
    """
    Test total energy calculation of `TensorAlloyCalculator` using the QM7
    dataset.
    """
    graph_path = join(test_dir(), 'checkpoints', 'qm7-k2', 'qm7.pb')
    sqlite3_file = join(datasets_dir(), 'qm7.db')

    calc = TensorAlloyCalculator(graph_path)

    db = connect(sqlite3_file)
    _, tests = train_test_split(np.arange(1, 1 + len(db)),
                                random_state=Defaults.seed, test_size=1000)

    y_true = []
    y_pred = []
    for index in tests:
        atoms = db.get_atoms(f'id={index}')
        y_true.append(atoms.get_total_energy())
        y_pred.append(calc.get_potential_energy(atoms))

    y_mae = mean_absolute_error(y_true, y_pred)
    assert_almost_equal(y_mae, 0.12429088, delta=1e-7)


if __name__ == "__main__":
    nose.run()
