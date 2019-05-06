# coding=utf-8
"""
This script is used to rebuild neighbor metadata of built-in datasets.
"""
from __future__ import print_function, absolute_import

from os.path import join

from tensoralloy.io.sqlite import connect
from tensoralloy.test_utils import datasets_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


built_in_datasets = {
    'qm7': {
        'rc': [6.5, 6.5],
        'k_max': [2, 3],
        'path': join(datasets_dir(), 'qm7.db'),
    },
    'snap-Ni': {
        'rc': [4.6, 4.6, 6.0, 6.5],
        'k_max': [2, 3, 2, 2],
        'path': join(datasets_dir(), 'snap-Ni.db'),
    },
    'snap-Mo': {
        'rc': [4.6, 4.6, 6.0, 6.5],
        'k_max': [2, 3, 2, 2],
        'path': join(datasets_dir(), 'snap-Mo.db'),
    },
    'snap': {
        'rc': [4.6, 4.6, 6.0, 6.5],
        'k_max': [2, 3, 2, 2],
        'path': join(datasets_dir(), 'snap.db'),
    }
}


def rebuild():
    """
    Rebuild metadata of built-in datasets.
    """
    for name, config in built_in_datasets.items():

        print(f"Dataset: {name} @ {config['path']}")
        db = connect(config['path'])

        for index in range(len(config['rc'])):
            rc = config['rc'][index]
            k_max = config['k_max'][index]

            if db.get_nij_max(k_max, rc, allow_calculation=False) is None:
                db.update_neighbor_meta(k_max, rc, verbose=True)
            else:
                print(f"Skip {name}/neighbor with rc={rc}, k_max={k_max}")

        if not db.get_atomic_static_energy(allow_calculation=False):
            db.get_atomic_static_energy(allow_calculation=True)
        else:
            print(f"Skip {name}/static_energy")


if __name__ == "__main__":
    rebuild()
