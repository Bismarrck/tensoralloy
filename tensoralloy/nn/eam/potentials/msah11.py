# coding=utf-8
"""
This module defines the empirical potential of Al-Fe proposed by Mendelev et al.
at 2011.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import functools
import warnings

from typing import List

from tensoralloy.nn.eam.potentials import EamFSPotential
from tensoralloy.nn.utils import log_tensor
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def _check_dtype():
    if get_float_dtype() == tf.float32:
        warnings.warn("It's not recommended to use float32 for MSAH11.")


class AlFeMsah11(EamFSPotential):
    """
    The Al-Fe proposed by Mendelev et al. at 2011.

    References
    ----------
    M.I. Mendelev, et al., J. Mater. Res. 20 (2011) 208.

    """

    def __init__(self):
        """
        Initialization method.
        """
        super(AlFeMsah11, self).__init__()

    @property
    def defaults(self):
        """
        The Msah11 potential does not have trainable parameters.
        """
        defaults = {"Al": {}, "Fe": {}}
        return defaults

    @staticmethod
    def _pairwise_func(lowcuts: np.ndarray, highcuts: np.ndarray,
                       c1: np.ndarray, c2: np.ndarray, coef: List[np.ndarray]):
        """
        Construct a pairwise function, phi(r), using the given parameters.
        """
        nfuncs = len(lowcuts)
        assert nfuncs == len(highcuts)
        assert nfuncs == len(coef) + 2

        def _poly(factors: np.ndarray, lowcut: float, highcut: float,
                  orders: np.ndarray, dtype=tf.float64):
            """
            A helper function to construct polynomial potentials.
            """
            def _func(r: tf.Tensor):
                lc = tf.convert_to_tensor(lowcut, name=f'lowcut', dtype=dtype)
                hc = tf.convert_to_tensor(highcut, name=f'highcut', dtype=dtype)
                idx = tf.where(
                    tf.logical_and(tf.greater_equal(r, lc), tf.less(r, hc)))
                x = tf.gather_nd(r, idx)
                shape = tf.shape(r, out_type=idx.dtype)
                values = []
                for i in range(len(factors)):
                    factor = tf.convert_to_tensor(factors[i], name=f'c{i}',
                                                  dtype=dtype)
                    order = tf.convert_to_tensor(orders[i], name=f'k{i}',
                                                 dtype=dtype)
                    y = factor * tf.pow(highcut - x, order)
                    values.append(y)
                return tf.scatter_nd(idx, tf.add_n(values), shape, name='poly')
            return _func

        def _first(factors: np.ndarray, lowcut: float, highcut: float,
                   dtype=tf.float64):
            """
            A helper function to construct the first potential term.
            """
            assert len(factors) % 2 == 1

            def _func(r: tf.Tensor):
                lc = tf.convert_to_tensor(lowcut, name=f'lowcut', dtype=dtype)
                hc = tf.convert_to_tensor(highcut, name=f'highcut', dtype=dtype)
                idx = tf.where(
                    tf.logical_and(tf.greater_equal(r, lc), tf.less(r, hc)))
                x = tf.gather_nd(r, idx)
                shape = tf.shape(r, out_type=idx.dtype)
                b = factors[1::2]
                c = factors[2::2]
                scale = tf.math.truediv(
                    tf.convert_to_tensor(factors[0], dtype=dtype), x)
                values = []
                for i in range(len(factors) // 2):
                    bi = tf.convert_to_tensor(b[i], dtype=dtype)
                    ci = tf.convert_to_tensor(c[i], dtype=dtype)
                    values.append(bi * tf.exp(ci * x))
                y = scale * tf.add_n(values)
                return tf.scatter_nd(idx, y, shape, name='first')
            return _func

        def _second(factors: np.ndarray, lowcut: float, highcut: float,
                    dtype=tf.float64):
            """
            A helper function to construct the second potential term.
            """
            assert len(factors) == 4

            def _func(r):
                lc = tf.convert_to_tensor(lowcut, name=f'lowcut', dtype=dtype)
                hc = tf.convert_to_tensor(highcut, name=f'highcut', dtype=dtype)
                idx = tf.where(
                    tf.logical_and(tf.greater_equal(r, lc), tf.less(r, hc)))
                x = tf.gather_nd(r, idx)
                shape = tf.shape(r, out_type=idx.dtype)
                c0 = tf.convert_to_tensor(factors[0], dtype=dtype)
                values = []
                for i in range(1, 4):
                    ci = tf.convert_to_tensor(factors[i], dtype=dtype)
                    values.append(ci * tf.pow(x, i))
                y = tf.exp(tf.add_n(values) + c0)
                return tf.scatter_nd(idx, y, shape, name='second')
            return _func

        def _phi(r: tf.Tensor):
            with tf.name_scope("First"):
                y1 = _first(
                    factors=c1, lowcut=lowcuts[0], highcut=highcuts[0],
                    dtype=r.dtype)(r)
            with tf.name_scope("Second"):
                y2 = _second(
                    factors=c2, lowcut=lowcuts[1], highcut=highcuts[1],
                    dtype=r.dtype)(r)

            values = [y1, y2]
            for i in range(2, nfuncs):
                with tf.name_scope(f"Poly{i}"):
                    yi = _poly(factors=coef[i - 2][:, 0], lowcut=lowcuts[i],
                               highcut=highcuts[i], orders=coef[i - 2][:, 1],
                               dtype=r.dtype)(r)
                    values.append(yi)
            return tf.add_n(values, name='phi')
        return _phi

    def phi(self, r: tf.Tensor, kbody_term: str, variable_scope: str,
            fixed=False, verbose=False):
        """
        The pairwise potential function.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.
        variable_scope : str
            The scope for variables of this potential function.
        fixed : bool
            Set all associated parameters to be fixed.
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        _check_dtype()

        with tf.name_scope("Mash11/Phi") as scope:
            r = tf.convert_to_tensor(r, name='r')
            np_dtype = r.dtype.as_numpy_dtype
            asarray = functools.partial(np.asarray, dtype=np_dtype)

            with tf.variable_scope(f"{scope}/{kbody_term}"):
                if kbody_term == 'AlAl':
                    highcuts = asarray([1.60, 2.25, 3.2, 4.8, 6.5])
                    lowcuts = asarray([1e-8, 1.6, 2.25, 2.25, 2.25])
                    c1 = asarray([
                        2433.5591473227,
                        0.1818, -22.713109144730,
                        0.5099, -6.6883008584622,
                        0.2802, -2.8597223982536,
                        0.02817, -1.4309258761180,
                    ])
                    c2 = asarray([
                        6.0801330531321, -2.3092752322555,
                        0.042696494305190, -0.07952189194038])
                    coef = [
                        asarray([
                            [17.222548257633, 4.0],
                            [-13.838795389103, 5.0],
                            [26.724085544227, 6.0],
                            [-4.8730831082596, 7.0],
                            [0.26111775221382, 8.0],
                        ]),
                        asarray([
                            [-1.8864362756631, 4.0],
                            [2.4323070821980, 5.0],
                            [-4.0022263154653, 6.0],
                            [1.3937173764119, 7.0],
                            [-0.31993486318965, 8.0],
                        ]),
                        asarray([
                            [0.30601966016455, 4.0],
                            [-0.63945082587403, 5.0],
                            [0.54057725028875, 6.0],
                            [-0.21210673993915, 7.0],
                            [0.03201431888287, 8.0],
                        ])
                    ]
                elif kbody_term == 'FeFe':
                    highcuts = asarray([1.0, 2.05, 2.2, 2.3, 2.4,
                                        2.5, 2.6, 2.7, 2.8, 3.0,
                                        3.3, 3.7, 4.2, 4.7, 5.3])
                    lowcuts = asarray([1e-8, 1.0, 2.05, 2.05, 2.05,
                                       2.05, 2.05, 2.05, 2.05, 2.05,
                                       2.05, 2.05, 2.05, 2.05, 2.05])
                    c1 = asarray([
                        9734.2365892908,
                        0.1818, -28.616724320005,
                        0.5099, -8.4267310396064,
                        0.2802, -3.6030244464156,
                        0.02817, -1.8028536321603,
                    ])
                    c2 = asarray([
                        7.4122709384068, -0.64180690713367,
                        -2.6043547961722, 0.62625393931230
                    ])
                    coef = [
                        asarray([[-27.444805994228, 3.0]]),
                        asarray([[15.738054058489, 3.0]]),
                        asarray([[2.2077118733936, 3.0]]),
                        asarray([[-2.4989799053251, 3.0]]),
                        asarray([[4.2099676494795, 3.0]]),
                        asarray([[-0.77361294129713, 3.0]]),
                        asarray([[0.80656414937789, 3.0]]),
                        asarray([[-2.3194358924605, 3.0]]),
                        asarray([[2.6577406128280, 3.0]]),
                        asarray([[-1.0260416933564, 3.0]]),
                        asarray([[0.35018615891957, 3.0]]),
                        asarray([[-0.058531821042271, 3.0]]),
                        asarray([[-0.0030458824556234, 3.0]]),
                    ]
                else:
                    highcuts = asarray([1.2, 2.2, 3.2, 6.2])
                    lowcuts = asarray([1e-8, 1.2, 2.2, 2.2])
                    c1 = asarray([
                        4867.1182946454,
                        0.1818, -25.834107666296,
                        0.5099, -7.6073373918597,
                        0.2802, -3.2526756183596,
                        0.02817, -1.6275487829767,
                    ])
                    c2 = asarray([
                        6.6167846784367, -1.5208197629514,
                        -0.73055022396300, -0.03879272494264,
                    ])
                    coef = [
                        asarray([
                            [-4.148701943924, 4.0],
                            [5.6697481153271, 5.0],
                            [-1.7835153896441, 6.0],
                            [-3.3886912738827, 7.0],
                            [1.9720627768230, 8.0],
                        ]),
                        asarray([
                            [0.094200713038410, 4.0],
                            [-0.16163849208165, 5.0],
                            [0.10154590006100, 6.0],
                            [-0.027624717063181, 7.0],
                            [0.0027505576632627, 8.0],
                        ])
                    ]
                comput = self._pairwise_func(lowcuts, highcuts, c1, c2, coef)
                phi = comput(r)
                if verbose:
                    log_tensor(phi)
                return phi

    def rho(self, r: tf.Tensor, kbody_term: str, variable_scope: str,
            fixed=False, verbose=False):
        """
        Return the Op to compute electron density `rho(r)`.

        Parameters
        ----------
        r : tf.Tensor
            A 5D tensor of shape `[batch_size, 1, max_n_element, nnl, 1]`.
        kbody_term : str
            The corresponding k-body term.
        variable_scope : str
            The scope for variables of this potential function.
        fixed : bool
            Set all associated parameters to be fixed.
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        _check_dtype()

        with tf.name_scope("Mash11/Rho") as scope:

            r = tf.convert_to_tensor(r, name='r')
            np_dtype = r.dtype.as_numpy_dtype
            asarray = functools.partial(np.asarray, dtype=np_dtype)

            def _density(_factors: np.ndarray, _cutoffs: np.ndarray,
                         _order: int):
                """
                A helper function.
                """
                zero = tf.constant(0.0, dtype=r.dtype, name='zero')
                values = []
                for i in range(len(_cutoffs)):
                    cutoff = tf.convert_to_tensor(_cutoffs[i], name=f'rc{i}',
                                                  dtype=r.dtype)
                    factor = tf.convert_to_tensor(_factors[i], name=f'c{i}',
                                                  dtype=r.dtype)
                    values.append(
                        factor * tf.pow(tf.maximum(cutoff - r, zero), _order))
                return tf.add_n(values, name='density')

            with tf.variable_scope(f"{scope}/{kbody_term}"):
                if kbody_term == 'AlAl':
                    factors = asarray([
                        0.00019850823042883, 0.10046665347629,
                        1.0054338881951E-01, 0.099104582963213,
                        0.090086286376778, 0.0073022698419468,
                        0.014583614223199, -0.0010327381407070,
                        0.0073219994475288, 0.0095726042919017])
                    cutoffs = asarray([2.5, 2.6, 2.7, 2.8, 3.0,
                                       3.4, 4.2, 4.8, 5.6, 6.5])
                    order = 4
                elif kbody_term == 'FeFe':
                    factors = asarray(
                        [11.686859407970, -0.014710740098830, 0.47193527075943])
                    cutoffs = asarray([2.4, 3.2, 4.2])
                    order = 3
                else:
                    factors = asarray([
                        0.010015421408039, 0.0098878643929526,
                        0.0098070326434207, 0.0084594444746494,
                        0.0038057610928282, -0.0014091094540309,
                        0.0074410802804324])
                    cutoffs = asarray([2.4, 2.5, 2.6, 2.8, 3.1,
                                       5.0, 6.2])
                    order = 4
                rho = _density(factors, cutoffs, order)
                if verbose:
                    log_tensor(rho)
                return rho

    def embed(self, rho: tf.Tensor, element: str, variable_scope: str,
              fixed=False, verbose=False):
        """
        Return the Op to compute the embedding energy F(rho(r)).

        Parameters
        ----------
        rho : tf.Tensor
            A 3D tensor of shape `[batch_size, max_n_element, 1]` where
            `max_n_element` is the maximum occurace of `element`.
        element : str
            An element symbol.
        variable_scope : str
            The scope for variables of this potential function.
        fixed : bool
            Set all associated parameters to be fixed.
        verbose : bool
            A bool. If True, key tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_elements]`.

        """
        _check_dtype()

        with tf.name_scope("Mash11/Embed") as scope:

            rho = tf.convert_to_tensor(rho, name='r')

            with tf.variable_scope(f"{scope}/{element}"):
                if element == 'Al':
                    c1 = tf.constant(
                        0.000093283590195398, dtype=rho.dtype, name='c1')
                    c2 = tf.constant(
                        0.0023491751192724, dtype=rho.dtype, name='c2')
                    eps = tf.constant(1e-12, dtype=rho.dtype, name='eps')
                    idx = tf.where(tf.greater_equal(rho, eps), name='idx')
                    shape = tf.shape(rho, out_type=idx.dtype)
                    x = tf.gather_nd(rho, idx)
                    y = -tf.sqrt(x) + c1 * tf.pow(x, 2) - c2 * x * tf.log(x)
                    embed = tf.scatter_nd(idx, y, shape, name='embed')
                else:
                    c3 = tf.constant(
                        0.00067314115586063, dtype=rho.dtype, name='c3')
                    c4 = tf.constant(
                        0.000000076514905604792, dtype=rho.dtype, name='c4')
                    y = tf.add(-tf.sqrt(rho),
                               -c3 * tf.pow(rho, 2) + c4 * tf.pow(rho, 4),
                               name='embed')
                    embed = y
                if verbose:
                    log_tensor(embed)
                return embed

    def dipole(self,
               r: tf.Tensor,
               kbody_term: str,
               variable_scope: str,
               fixed=False,
               verbose=False):
        """
        AlFeMsah11 does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have dipole term.")

    def quadrupole(self,
                   r: tf.Tensor,
                   kbody_term: str,
                   variable_scope: str,
                   fixed=False,
                   verbose=False):
        """
        AlFeMsah11 does not support calculating dipole.
        """
        raise Exception(
            f"{self.__class__.__name__} does not have quadrupole term.")
