# coding=utf-8
"""
This script is used to rebuild neighbor metadata of built-in datasets.
"""
from __future__ import print_function, absolute_import

from ase.db import connect
from os.path import join

from tensoralloy.io.neighbor import find_neighbor_size_maximums
from tensoralloy.io.neighbor import convert_k_max_to_key
from tensoralloy.io.neighbor import convert_rc_to_key
from tensoralloy.dataset.utils import compute_atomic_static_energy
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
            rc_key = convert_rc_to_key(rc)
            k_max_key = convert_k_max_to_key(k_max)

            try:
                _ = db.metadata['neighbors'][k_max_key][rc_key]['nnl_max']
            except KeyError:
                find_neighbor_size_maximums(db, rc, k_max, verbose=True)
            else:
                print(f"Skip {name}/neighbor with rc={rc_key}, "
                      f"k_max={k_max_key}")

            if 'atomic_static_energy' not in db.metadata:
                elements = sorted(list(db.metadata['max_occurs'].keys()))
                compute_atomic_static_energy(db, elements, verbose=True)
            else:
                print(f"Skip {name}/static_energy")

        print('Done.\n')


if __name__ == "__main__":
    rebuild()
