#!coding=utf-8
"""
Cubic spline based EAM/ADP potentials.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json

from tensoralloy.nn.eam.potentials.potentials import EamAlloyPotential
from tensoralloy.nn.eam.potentials.potentials import get_variable
from tensoralloy.nn.utils import log_tensor
from tensoralloy.extension.interp.cubic import CubicInterpolator

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class CubicSplinePotential(EamAlloyPotential):
    """
    A general cubic spline based EAM/ADP potential container.
    """

    def __init__(self, filename: str):
        """
        Initialization method.

        Parameters
        ----------
        filename : str
            The initial guess file. This should be a csv file without index.

        """
        super(CubicSplinePotential, self).__init__()
        self._name = "Spline"

        with open(filename) as fp:
            self._df = dict(json.load(fp))

    @property
    def defaults(self):
        """
        The default parameters.
        """
        return {}

    def is_avaiblable(self, key):
        """
        Return True if the potential is available.
        """
        keys = self._df.keys()
        return f"{key}.x" in keys and f"{key}.y" in keys

    def _make(self, t: tf.Tensor, pot: str, name: str):
        """
        Make a spline potential.
        """
        if not self.is_avaiblable(pot):
            raise ValueError(f"{pot} is not available!")

        dtype = t.dtype
        xval = np.asarray(self._df[f"{pot}.x"], dtype=dtype.as_numpy_dtype)
        yval = np.asarray(self._df[f"{pot}.y"], dtype=dtype.as_numpy_dtype)
        x = tf.convert_to_tensor(xval, dtype=dtype, name="x")
        y = get_variable(
            "y", shape=yval.shape, dtype=dtype, trainable=True,
            initializer=tf.constant_initializer(yval, dtype=dtype),
            collections=[tf.GraphKeys.MODEL_VARIABLES, ]
        )
        bc_start = self._df[f"{pot}.bc_start"]
        bc_end = self._df[f"{pot}.bc_end"]
        natural_boundary = self._df[f"{pot}.natural_boundary"]
        cubic = CubicInterpolator(x, y, natural_boundary=natural_boundary,
                                  bc_start=bc_start, bc_end=bc_end)
        shape = tf.shape(t, name='shape')
        val = cubic.run(tf.reshape(t, (-1,), name='t/flat'), name='eval')
        val = tf.reshape(val, shape, name=name)
        return val

    def rho(self, r: tf.Tensor, element: str, variable_scope: str,
            verbose=False):
        """
        The electron density function rho(r).

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, max_n_terms, 1, nnl, 1]`.
        element : str
            The corresponding element.
        variable_scope : str
            The scope for variables of this potential function.
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        with tf.name_scope(f"{self._name}/Rho/{element}"):
            with tf.variable_scope(
                    f"{variable_scope}/{element}",
                    reuse=tf.AUTO_REUSE):
                rho = self._make(r, f"{element}.rho", "rho")
                if verbose:
                    log_tensor(rho)
                return rho

    def embed(self,
              rho: tf.Tensor,
              element: str,
              variable_scope: str,
              verbose=False):
        """
        The embedding function.
        """
        with tf.name_scope(f"{self._name}/Embed/{element}"):
            with tf.variable_scope(
                    f"{variable_scope}/{element}",
                    reuse=tf.AUTO_REUSE):
                embed = self._make(rho, f"{element}.embed", "embed")
                if verbose:
                    log_tensor(embed)
                return embed

    def phi(self,
            r: tf.Tensor,
            kbody_term: str,
            variable_scope: str,
            verbose=False):
        """
        The pairwise interaction function.
        """
        with tf.name_scope(f"{self._name}/Phi/{kbody_term}"):
            with tf.variable_scope(
                    f"{variable_scope}/{kbody_term}",
                    reuse=tf.AUTO_REUSE):
                phi = self._make(r, f"{kbody_term}.phi", "phi")
                if verbose:
                    log_tensor(phi)
                return phi

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               verbose=False):
        """
        The dipole function.
        """
        with tf.name_scope(f"{self._name}/Dipole/{kbody_term}"):
            with tf.variable_scope(
                    f"{variable_scope}/{kbody_term}",
                    reuse=tf.AUTO_REUSE):
                dipole = self._make(r, f"{kbody_term}.dipole", "u")
                if verbose:
                    log_tensor(dipole)
                return dipole

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   verbose=False):
        """
        The quadrupole function.
        """
        with tf.name_scope(f"{self._name}/Quadrupole/{kbody_term}"):
            with tf.variable_scope(
                    f"{variable_scope}/{kbody_term}",
                    reuse=tf.AUTO_REUSE):
                quadrupole = self._make(r, f"{kbody_term}.quadrupole", "w")
                if verbose:
                    log_tensor(quadrupole)
                return quadrupole

    def gs(self,
           r: tf.Tensor,
           kbody_term: str,
           variable_scope: str,
           verbose=False):
        """
        The Gs function for meam/spline.
        """
        with tf.name_scope(f"{self._name}/Gs/{kbody_term}"):
            with tf.variable_scope(
                    f"{variable_scope}/{kbody_term}",
                    reuse=tf.AUTO_REUSE):
                gs = self._make(r, f"{kbody_term}.gs", "w")
                if verbose:
                    log_tensor(gs)
                return gs

    def fs(self,
           r: tf.Tensor,
           element: str,
           variable_scope: str,
           verbose=False):
        """
        The Fs function for meam/spline.
        """
        with tf.name_scope(f"{self._name}/Fs/{element}"):
            with tf.variable_scope(
                    f"{variable_scope}/{element}",
                    reuse=tf.AUTO_REUSE):
                fs = self._make(r, f"{element}.fs", "w")
                if verbose:
                    log_tensor(fs)
                return fs


class LinearlyExtendedSplinePotential(CubicSplinePotential):
    """
    A special spline potential for pair style 'meam/spline'.

    When x < xmin: f(x) = y[0] + (x - xmin) * deriv0
    When x > xmax: f(x) = y[-1] + (x - xmax) * deriv1

    """

    def _make(self, t: tf.Tensor, pot: str, name: str):
        """
        Make a spline potential.
        """
        if not self.is_avaiblable(pot):
            raise ValueError(f"{pot} is not available!")
        if self._df[f"{pot}.natural_boundary"]:
            raise ValueError("Natural boundary is not applicable for "
                             "LinearlyExtendedSplinePotential")

        dtype = t.dtype
        xval = np.asarray(self._df[f"{pot}.x"], dtype=dtype.as_numpy_dtype)
        yval = np.asarray(self._df[f"{pot}.y"], dtype=dtype.as_numpy_dtype)
        x = tf.convert_to_tensor(xval, dtype=dtype, name="x")
        y = get_variable(
            "y",
            shape=yval.shape,
            dtype=dtype,
            trainable=True,
            initializer=tf.constant_initializer(yval, dtype=dtype),
            collections=[tf.GraphKeys.MODEL_VARIABLES, ]
        )
        deriv0 = tf.constant(
            self._df[f"{pot}.bc_start"], dtype=dtype, name='deriv0')
        derivN = tf.constant(
            self._df[f"{pot}.bc_end"], dtype=dtype, name='derivN')
        cubic = CubicInterpolator(x, y, natural_boundary=False,
                                  bc_start=deriv0, bc_end=derivN)
        original_shape = tf.shape(t, name='shape')
        t = tf.reshape(t, (-1,), name='t/flat')

        with tf.name_scope("Seg"):
            idx1 = tf.where(tf.less(t, x[0]), name='idx1')
            idx2 = tf.where(tf.logical_and(tf.greater_equal(t, x[0]),
                                           tf.less(t, x[-1])), name='idx2')
            idx3 = tf.where(tf.greater_equal(t, x[-1]), name='idx3')
            shape = tf.shape(t, name='shape', out_type=idx1.dtype)

            def left_fn(z):
                """
                The linear function for x < xmin
                """
                return tf.math.add(y[0],
                                   deriv0 * tf.math.subtract(z, x[0]),
                                   name='left')

            def mid_fn(z):
                """
                The cubic spline function for xmin <= x <= xmax
                """
                return cubic.run(z, name='mid')

            def right_fn(z):
                """
                The linear function for x > xmax
                """
                return tf.math.add(y[-1],
                                   derivN * (tf.math.subtract(z, x[-1])),
                                   name='right')

            values = [
                tf.scatter_nd(idx1, left_fn(tf.gather_nd(t, idx1)), shape),
                tf.scatter_nd(idx2, mid_fn(tf.gather_nd(t, idx2)), shape),
                tf.scatter_nd(idx3, right_fn(tf.gather_nd(t, idx3)), shape),
            ]
            val = tf.add_n(values, name='eval')

        val = tf.reshape(val, original_shape, name=name)
        return val
