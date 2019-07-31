#!coding=utf-8
"""
Cubic spline based EAM/ADP potentials.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import pandas as pd

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
        self._df = pd.read_csv(filename, index_col=None)

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

        nr = self._adpfl.nr
            x = self._adpfl.rho[element][0][0:nr:self._interval]
            y = self._adpfl.rho[element][1][0:nr:self._interval]
            f = CubicInterpolator(x, y, natural_boundary=True, name='Spline')
            shape = tf.shape(r, name='shape')
            rho = f.evaluate(tf.reshape(r, (-1,), name='r/flat'))
            rho = tf.reshape(rho, shape, name='rho')
            if verbose:
                log_tensor(rho)
            return rho

        """
        if not self.is_avaiblable(pot):
            raise ValueError(f"{pot} is not available!")

        dtype = t.dtype
        xval = self._df[f"{pot}.x"].to_numpy(dtype.as_numpy_dtype)
        yval = self._df[f"{pot}.y"].to_numpy(dtype.as_numpy_dtype)
        x = tf.convert_to_tensor(xval, dtype=dtype, name="x")
        y = get_variable(
            "y", shape=yval.shape, dtype=dtype, trainable=True,
            initializer=tf.constant_initializer(yval, dtype=dtype),
            collections=[tf.GraphKeys.MODEL_VARIABLES, ]
        )
        cubic = CubicInterpolator(x, y, natural_boundary=True)
        shape = tf.shape(t, name='shape')
        val = cubic.evaluate(tf.reshape(t, (-1, ), name='t/flat'), name='eval')
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
                embed = self._make(rho, f"{element}.embed", "phi")
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
