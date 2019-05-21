# -*- coding: utf-8 -*-
"""
The TensorFlow based cubic spline implementation.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow.python.framework import ops

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

    def __init__(self, x, y, bc_start=None, bc_end=None, natural_boundary=False):
        """
        Initialization method.

        Parameters
        ----------
        x : tf.Tensor
            The independent coordinates of the training points. The shape should
            be `[..., N]`.
        y : tf.Tensor
            The dependent coordinates of the training points. This must be the
            same shape as ``x`` and the interpolation is always performed along
            the last axis. The shape should be `[..., N]`.
        bc_start : float or tf.Tensor or None
            The value of the derivative of the function at the first data point.
            By default this is zero.
        bc_end : float or tf.Tensor or None
            The value of the derivative of the function at the last data point.
            By default this is zero.

        """
        self.x = x
        self.y = y
        self.bc_start = bc_start
        self.bc_end = bc_end
        self.use_natural_boundary = natural_boundary

    def evaluate(self, t, name=None):
        """
        Interpolate the training points using a cubic spline.

        Parameters
        ----------
        t : tf.Tensor
            The independent coordinates where the model should be evaluated.
            The dimensions of all but the first axis must match the last
            dimension of `x`. The interpolation is performed in the last
            dimension independently for each of the earlier dimensions.
            The shape should be `[N, ...]`.
        name : str
            The name.

        """
        with ops.name_scope(name, "CubicInterpolator", [t]):

            x = tf.convert_to_tensor(self.x, name='x')
            y = tf.convert_to_tensor(self.y, name='y')

            # Compute the deltas
            size = tf.shape(x)[-1]
            axis = tf.rank(x) - 1
            dx = tf.math.subtract(tf.gather(x, tf.range(1, size), axis=axis),
                                  tf.gather(x, tf.range(size - 1), axis=axis),
                                  name='dx')
            dy = tf.math.subtract(tf.gather(y, tf.range(1, size), axis=axis),
                                  tf.gather(y, tf.range(size - 1), axis=axis),
                                  name='dy')

            # Compute the slices
            upper_inds = tf.range(1, size - 1)
            lower_inds = tf.range(size - 2)
            s_up = lambda a: tf.gather(a, upper_inds, axis=axis)  # NOQA
            s_lo = lambda a: tf.gather(a, lower_inds, axis=axis)  # NOQA
            dx_up = s_up(dx)
            dx_lo = s_lo(dx)
            dy_up = s_up(dy)
            dy_lo = s_lo(dy)

            first = lambda a: tf.gather(a, tf.zeros(1, dtype=tf.int64),  # NOQA
                                        axis=axis)
            last = lambda a: tf.gather(a, [size - 2], axis=axis)  # NOQA

            if self.bc_start is None:
                bc_start = 0.0
            bc_start = tf.convert_to_tensor(
                bc_start, dtype=x.dtype, name='bc_start')

            if self.bc_end is None:
                bc_end = 0.0
            bc_end = tf.convert_to_tensor(bc_end, dtype=x.dtype, name='bc_end')

            two = tf.constant(2.0, dtype=x.dtype, name='two')
            three = tf.constant(3.0, dtype=x.dtype, name='three')

            if not self.use_natural_boundary:
                upper = dx
                lower = dx
                diag = two * tf.concat((first(dx), dx_up + dx_lo, last(dx)),
                                       axis=axis)
                Y = three * tf.concat((first(dy) / first(dx) - bc_start,
                                       dy_up / dx_up - dy_lo / dx_lo,
                                       bc_end - last(dy) / last(dx)),
                                      axis=axis, name='y')

            else:
                diag = two * tf.concat((first(dx) / two,
                                        dx_up + dx_lo,
                                        last(dx) / two),
                                       axis=axis, name='diag')

                shape = dx.shape.as_list()[:-1] + [1]
                zeros = tf.zeros(shape, dtype=x.dtype)
                upper = tf.concat((zeros, dx[..., 1:]),
                                  axis=axis, name='upper')
                lower = tf.concat((dx[..., :-1], zeros),
                                  axis=axis, name='lower')
                Y = three * tf.concat((tf.fill(shape, bc_start),
                                       dy_up / dx_up - dy_lo / dx_lo,
                                       tf.fill(shape, bc_end)),
                                      axis=axis, name=name)

            # Solve the tri-diagonal system Ax = b
            c = tri_diag_solve(diag, upper, lower, Y)
            c_up = tf.gather(c, tf.range(1, size), axis=axis)
            c_lo = tf.gather(c, tf.range(size - 1), axis=axis)
            b = dy / dx - dx * (c_up + two * c_lo) / three
            d = (c_up - c_lo) / (three * dx)

            res = cubic_op.cubic_gather(t, x, y, b, c_lo, d)
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
