# -*- coding: utf-8 -*-
"""
The TensorFlow based cubic spline implementation.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensoralloy.extension.interp.utils import load_op_library
from tensoralloy.extension.interp.solver import tri_diag_solve


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = [
    "CubicInterpolator",
]


cubic_op = load_op_library("cubic_op")


class CubicInterpolator(object):
    """
    Cubic spline interpolation for a scalar function in one dimension
    """

    def __init__(self, x, y, fpa=None, fpb=None, name=None):
        """
        Initialization method.

        Parameters
        ----------
        x : tf.Tensor
            The independent coordinates of the training points.
        y : tf.Tensor
            The dependent coordinates of the training points. This must be the
            same shape as ``x`` and the interpolation is always performed along
            the last axis.
        fpa : float or tf.Tensor or None
            The value of the derivative of the function at the first data point.
            By default this is zero.
        fpb : float or tf.Tensor or None
            The value of the derivative of the function at the last data point.
            By default this is zero.
        name : str
            The name of this interpolator.
        """

        self.name = name
        with tf.name_scope(name, "CubicInterpolator"):
            # Compute the deltas
            size = tf.shape(x)[-1]
            axis = tf.rank(x) - 1
            dx = tf.gather(x, tf.range(1, size), axis=axis) \
                - tf.gather(x, tf.range(size-1), axis=axis)
            dy = tf.gather(y, tf.range(1, size), axis=axis) \
                - tf.gather(y, tf.range(size-1), axis=axis)

            # Compute the slices
            upper_inds = tf.range(1, size-1)
            lower_inds = tf.range(size-2)
            s_up = lambda a: tf.gather(a, upper_inds, axis=axis)  # NOQA
            s_lo = lambda a: tf.gather(a, lower_inds, axis=axis)  # NOQA
            dx_up = s_up(dx)
            dx_lo = s_lo(dx)
            dy_up = s_up(dy)
            dy_lo = s_lo(dy)

            first = lambda a: tf.gather(a, tf.zeros(1, dtype=tf.int64),  # NOQA
                                        axis=axis)
            last = lambda a: tf.gather(a, [size-2], axis=axis)  # NOQA

            fpa_ = fpa if fpa is not None else tf.constant(0, x.dtype)
            fpb_ = fpb if fpb is not None else tf.constant(0, x.dtype)

            diag = 2 * tf.concat((first(dx), dx_up+dx_lo, last(dx)), axis)
            upper = dx
            lower = dx
            Y = 3*tf.concat((first(dy)/first(dx) - fpa_,
                             dy_up/dx_up - dy_lo/dx_lo,
                             fpb_ - last(dy)/last(dx)), axis)

            # Solve the tri-diagonal system
            c = tri_diag_solve(diag, upper, lower, Y)
            c_up = tf.gather(c, tf.range(1, size), axis=axis)
            c_lo = tf.gather(c, tf.range(size-1), axis=axis)
            b = dy / dx - dx * (c_up + 2*c_lo) / 3
            d = (c_up - c_lo) / (3*dx)

            self.x = x
            self.y = y
            self.b = b
            self.c = c_lo
            self.d = d

    def evaluate(self, t, name=None):
        """
        Interpolate the training points using a cubic spline.

        Parameters
        ----------
        t : tf.Tensor
            The independent coordinates where the model should be evaluated.
            The dimensions of all but the last axis must match the
            dimensions of ``x``. The interpolation is performed in the last
            dimension independently for each of the earlier dimensions.
        name : str
            The name.

        """
        with tf.name_scope(self.name, "CubicInterpolator"):
            with tf.name_scope(name, "evaluate"):
                res = cubic_op.cubic_gather(t, self.x, self.y, self.b, self.c,
                                            self.d)
                tau = t - res.xk
                mod = res.ak + res.bk * tau + res.ck * tau**2 + res.dk * tau**3
                return mod


@tf.RegisterGradient("CubicGather")
def _cubic_gather_rev(op, *grads):
    x = op.inputs[1]
    inds = op.outputs[-1]
    args = [x, inds] + list(grads)
    results = cubic_op.cubic_gather_rev(*args)
    return [tf.zeros_like(op.inputs[0])] + list(results)
