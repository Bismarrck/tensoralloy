#!coding=utf-8
"""
Generic database functions.
"""
from __future__ import print_function, absolute_import

import os

from os.path import splitext
from pathlib import PurePath
from ase.db import connect as ase_connect
from ase.parallel import world

from tensoralloy.io.sqlite import CoreDatabase


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["CoreDatabase", "connect"]


def connect(name, use_lock_file=True, append=True, serial=False):
    """
    Create connection to database.

    Parameters
    ----------
    name: str
        Filename or address of database.
    use_lock_file: bool
        You can turn this off if you know what you are doing ...
    append: bool
        Use append=False to start a new database.
    serial : bool
        Let someone else handle parallelization.  Default behavior is to
        interact with the database on the master only and then distribute
        results to all slaves.

    """
    if isinstance(name, PurePath):
        name = str(name)

    if not append and world.rank == 0:
        if isinstance(name, str) and os.path.isfile(name):
            os.remove(name)

    if name is None:
        db_type = None
    elif not isinstance(name, str):
        db_type = 'json'
    elif (name.startswith('postgresql://') or
          name.startswith('postgres://')):
        db_type = 'postgresql'
    else:
        db_type = splitext(name)[1][1:]
        if db_type == '':
            raise ValueError('No file extension or database type given')

    if db_type == 'db':
        return CoreDatabase(name,
                            create_indices=True,
                            use_lock_file=use_lock_file,
                            serial=serial)
    else:
        return ase_connect(name,
                           type=db_type,
                           create_indices=True,
                           use_lock_file=use_lock_file,
                           append=append,
                           serial=serial)
