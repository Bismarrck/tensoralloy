# coding=utf-8
"""
This module defines utility functions for constructing datasets.
"""
from __future__ import print_function, absolute_import

import re
import platform

from subprocess import PIPE, Popen

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def brange(start, stop, batch_size):
    """
    Range from `start` to `stop` given a batch size and return the start and
    stop of each batch.

    Parameters
    ----------
    start : int
        The start number of a sequence.
    stop : int,
        The end number of a sequence.
    batch_size : int
        The size of each batch.

    Yields
    ------
    istart : int
        The start number of a batch.
    istop : int
        The end number of a batch.

    """
    istart = start
    while istart < stop:
        istop = min(istart + batch_size, stop)
        yield istart, istop
        istart = istop


def should_be_serial():
    """
    Return True if the dataset should be in serial mode.

    For macOS this function always return False.
    For Linux if `glibc>=2.17` return False; otherwise return True.

    """
    if platform.system() == 'Linux':
        pattern = re.compile(r'^GLIBC_2.([\d.]+)')
        p = Popen('strings /lib64/libc.so.6 | grep GLIBC_2.',
                  shell=True, stdout=PIPE, stderr=PIPE)
        (stdout, stderr) = p.communicate()
        if stderr.decode('utf-8') != '':
            return True
        glibc_ver = 0.0
        for line in stdout.decode('utf-8').split('\n'):
            m = pattern.search(line)
            if m:
                glibc_ver = max(float(m.group(1)), glibc_ver)
        if glibc_ver >= 17.0:
            return False
        else:
            return True
    else:
        return False

