# coding=utf-8
"""
This module defines various cutoff fucntions.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["cosine_cutoff", "polynomial_cutoff", "meam_cutoff", "deepmd_cutoff",
           "tersoff_cutoff"]


def cosine_cutoff(r: tf.Tensor, rc: float, name=None):
    """
    The cosine cutoff function proposed by Jörg Behler:

        fc(r) = 0.5 * [ cos(min(r / rc) * pi) + 1 ]

    Parameters
    ----------
    r : tf.Tensor
        A `float64` or `float32` tensor.
    rc : float or tf.Tensor
        The cutoff radius. fc(r) = 0 if r > rc.
    name : str or None
        The name of this Op.

    See Also
    --------
    PRL 98, 146401 (2007)

    """
    with ops.name_scope(name, "CosCutoff", [r, rc]) as name:
        r = ops.convert_to_tensor(r, name='r')
        rc = ops.convert_to_tensor(rc, dtype=r.dtype, name="rc")
        ratio = math_ops.truediv(r, rc, name='ratio')
        one = ops.convert_to_tensor(1.0, dtype=r.dtype, name='one')
        half = ops.convert_to_tensor(0.5, dtype=r.dtype, name='half')
        z = math_ops.minimum(ratio, one, name='minimum')
        z = math_ops.cos(z * np.pi, name='cos') + one
        return math_ops.multiply(z, half, name=name)


def polynomial_cutoff(r: tf.Tensor, rc: float, gamma=5.0, name=None):
    """
    The polynomial cutoff function proposed by Andrew Peterson:

        fc(r) = 1 + gamma * (r / rc)^(gamma + 1) - (gamma + 1) * (r / rc)^gamma

    Parameters
    ----------
    r : tf.Tensor
        A `float64` or `float32` tensor.
    rc : float or tf.Tensor
        The cutoff radius. fc(r) = 0 if r > rc.
    gamma : float or tf.Tensor
        The hyper parameter of this Op. The polynomial function approximates the
        cosine cutoff function’s behavior if `gamma` is set to 2; as `gamma` is
        increased the decay is delayed, but the limiting behaviors are intact.
        As `gamma` is increased to very large values, it approaches the step
        function, which would be equivalent to not employing a cutoff function.
    name : str or None
        The name of this Op.

    See Also
    --------
    Comput. Phys. Commun. 207 (2016) 310–324

    """
    with ops.name_scope(name, "PolyCutoff", [r, rc, gamma]) as name:
        r = ops.convert_to_tensor(r, name='r')
        gamma = ops.convert_to_tensor(gamma, dtype=r.dtype, name='gamma')
        rc = ops.convert_to_tensor(rc, dtype=r.dtype, name="rc")
        one = ops.convert_to_tensor(1.0, dtype=r.dtype, name='one')
        div = math_ops.truediv(r, rc, name='ratio')
        div = math_ops.minimum(one, div)
        z = gamma * div**(gamma + one) - (gamma + one) * div**gamma
        return tf.add(z, one, name=name)


def meam_cutoff(x: tf.Tensor, name=None):
    """
    The MEAM cutoff function.
                1                       x >= 1
        f(x) =  (1 - (1 - x)**4)**2     0 < x < 1
                0                       x <= 0

    """
    with ops.name_scope(name, "MeamCutoff", [x]) as name:
        one = ops.convert_to_tensor(1.0, dtype=x.dtype, name='one')
        x = tf.nn.relu(x)
        x = tf.minimum(x, one)
        return tf.square(one - (one - x)**4, name=name)


def deepmd_cutoff(r: tf.Tensor, rc: float, rcs: float, name=None):
    """
    The DeePMD cutoff function.
    """
    with ops.name_scope(name, "DeepmdCutoff", [r, rc, rcs]) as name:
        r = ops.convert_to_tensor(r, name='r')
        rc = ops.convert_to_tensor(rc, dtype=r.dtype, name='rc')
        rcs = ops.convert_to_tensor(rcs, dtype=r.dtype, name='rcs')
        half = ops.convert_to_tensor(0.5, dtype=r.dtype, name='half')
        one = ops.convert_to_tensor(1.0, dtype=r.dtype, name='one')
        zero = ops.convert_to_tensor(0.0, dtype=r.dtype, name='zero')
        recip = math_ops.div_no_nan(one, r, name='recip')
        step = math_ops.subtract(rc, rcs)
        dr = math_ops.subtract(r, rcs)
        z = math_ops.divide(dr, step, name='z')
        z = math_ops.maximum(z, zero, name='zmax')
        z = math_ops.minimum(z, one, name='zmin')
        pi = ops.convert_to_tensor(np.pi, dtype=r.dtype, name='pi')
        zpi = math_ops.multiply(pi, z)
        return math_ops.multiply(recip, half * math_ops.cos(zpi) + half,
                                 name=name)


def tersoff_cutoff(r: tf.Tensor, R: float, D: float, name=None):
    """
    The Tersoff cutoff function.
    """
    with ops.name_scope(name, "TersoffCutoff", [r, R, D]) as name:
        R = ops.convert_to_tensor(R, dtype=r.dtype, name='R')
        D = ops.convert_to_tensor(D, dtype=r.dtype, name='D')
        lb = ops.convert_to_tensor(-1.0, dtype=r.dtype, name='lb')
        ub = ops.convert_to_tensor(1.0, dtype=r.dtype, name='ub')
        pi_h = ops.convert_to_tensor(np.pi / 2.0, dtype=r.dtype, name='pi')
        half = ops.convert_to_tensor(0.5, dtype=r.dtype, name='half')
        diff = math_ops.subtract(r, R, name='diff')
        z = math_ops.truediv(diff, D)
        z = math_ops.minimum(math_ops.maximum(lb, z), ub)
        z = math_ops.subtract(
            half,
            math_ops.multiply(half, math_ops.sin(pi_h * z)), name=name)
        return z
