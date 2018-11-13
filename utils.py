# coding=utf-8
"""
This module defines tensorflow-based functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import logging
from logging.config import dictConfig
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def cutoff(r: tf.Tensor, rc: float, name=None):
    """
    The cutoff function.

    f_c(r) = 0.5 * [ cos(min(r / rc) * pi) + 1 ]

    """
    with ops.name_scope(name, "fc", [r]) as name:
        rc = ops.convert_to_tensor(rc, dtype=r.dtype, name="rc")
        ratio = math_ops.div(r, rc, name='ratio')
        z = math_ops.minimum(ratio, 1.0, name='minimum')
        z = math_ops.cos(z * np.pi, name='cos') + 1.0
        return math_ops.multiply(z, 0.5, name=name)


def set_logging_configs(logfile="logfile"):
    """
    Setup the logging module.
    """
    LOGGING_CONFIG = {
        "version": 1,
        "formatters": {
            'file': {
                'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
            },
        },
        "handlers": {
            'file': {
                'class': 'logging.FileHandler',
                'level': logging.INFO,
                'formatter': 'file',
                'filename': logfile,
                'mode': 'a',
            },
        },
        "root": {
            'handlers': ['file'],
            'level': logging.INFO,
        },
        "disable_existing_loggers": False
    }
    dictConfig(LOGGING_CONFIG)
