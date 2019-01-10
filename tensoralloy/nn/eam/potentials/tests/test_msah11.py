# coding=utf-8
"""
This module defines tests for the MSAH11 Al-Fe potential.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import math
import nose

from tensoralloy.nn.eam.potentials.msah11 import AlFeMsah11
from tensoralloy.test_utils import assert_array_almost_equal
from tensoralloy.dtypes import get_float_dtype, set_float_precision, Precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def atsim_embed_al(rho):
    """
    The embedding function of Al by ATSIM.
    """
    if rho == 0.0:
        return 0.0
    y1 = -math.sqrt(rho)
    y2 = 0.000093283590195398 * rho ** 2
    y3 = - 0.0023491751192724 * rho * math.log(rho)
    return y1 + y2 + y3


def atsim_embed_fe(rho):
    """
    The embedding function of Fe by ATSIM.
    """
    y1 = -math.sqrt(rho)
    y2 = - 0.00067314115586063 * rho ** 2
    y3 = 0.000000076514905604792 * rho ** 4
    return y1 + y2 + y3


def atsim_density_alal(r):
    """
    The density function of Al-Al by ATSIM.
    """
    funcs = [
        (2.5, lambda x: 0.00019850823042883 * (2.5 - x) ** 4),
        (2.6, lambda x: 0.10046665347629 * (2.6 - x) ** 4),
        (2.7, lambda x: 1.0054338881951E-01 * (2.7 - x) ** 4),
        (2.8, lambda x: 0.099104582963213 * (2.8 - x) ** 4),
        (3.0, lambda x: 0.090086286376778 * (3.0 - x) ** 4),
        (3.4, lambda x: 0.0073022698419468 * (3.4 - x) ** 4),
        (4.2, lambda x: 0.014583614223199 * (4.2 - x) ** 4),
        (4.8, lambda x: -0.0010327381407070 * (4.8 - x) ** 4),
        (5.6, lambda x: 0.0073219994475288 * (5.6 - x) ** 4),
        (6.5, lambda x: 0.0095726042919017 * (6.5 - x) ** 4)]
    vals = [func(r) for cutoff, func in funcs if r <= cutoff]
    return sum(vals)


def atsim_density_fefe(r):
    """
    The density function of Fe-Fe by ATSIM.
    """
    funcs = [
        (2.4, lambda x: 11.686859407970 * (2.4 - x) ** 3),
        (3.2, lambda x: -0.014710740098830 * (3.2 - x) ** 3),
        (4.2, lambda x: 0.47193527075943 * (4.2 - x) ** 3)]
    vals = [func(r) for cutoff, func in funcs if r <= cutoff]
    return sum(vals)


def atsim_density_alfe(r):
    """
    The density function of Fe-Al and Al-Fe by ATSIM.

    Notes
    -----
    Although both the αβ and βα can be described using eam/fs files, the
    Mendelev model used in this example uses the same density function for both
    Al-Fe and Fe-Al cross density functions.

    """
    funcs = [
        (2.4, lambda x: 0.010015421408039 * (2.4 - x) ** 4),
        (2.5, lambda x: 0.0098878643929526 * (2.5 - x) ** 4),
        (2.6, lambda x: 0.0098070326434207 * (2.6 - x) ** 4),
        (2.8, lambda x: 0.0084594444746494 * (2.8 - x) ** 4),
        (3.1, lambda x: 0.0038057610928282 * (3.1 - x) ** 4),
        (5.0, lambda x: -0.0014091094540309 * (5.0 - x) ** 4),
        (6.2, lambda x: 0.0074410802804324 * (6.2 - x) ** 4)]
    vals = [func(r) for cutoff, func in funcs if r <= cutoff]
    return sum(vals)


def zerowrap(wrapped):
    """
    A helper function.
    """
    def _func(r):
        if r == 0.0:
            return 0.0
        return wrapped(r)
    return _func


def atsim_pairwise_alal(r):
    """
    The pairwise potential of Al-Al.
    """
    funcs = [
        ((0.0, 1.60),
         zerowrap(lambda x: (2433.5591473227 / x) *
                            (0.1818 * math.exp(-22.713109144730 * x) +
                             0.5099 * math.exp(-6.6883008584622 * x) +
                             0.2802 * math.exp(-2.8597223982536 * x) +
                             0.02817 * math.exp(-1.4309258761180 * x)))),
        ((1.6, 2.25),
         lambda x: math.exp(6.0801330531321 - 2.3092752322555 * x +
                            0.042696494305190 * x**2 -
                            0.07952189194038 * x**3)),
        ((2.25, 3.2),
         lambda x: (17.222548257633 * (3.2 - x)**4 -
                    13.838795389103 * (3.2 - x)**5 +
                    26.724085544227 * (3.2 - x)**6 -
                    4.8730831082596 * (3.2 - x)**7 +
                    0.26111775221382 * (3.2 - x)**8)),
        ((2.25, 4.8),
         lambda x: (-1.8864362756631 * (4.8 - x)**4 +
                    2.4323070821980 * (4.8 - x)**5 -
                    4.0022263154653 * (4.8 - x)**6 +
                    1.3937173764119 * (4.8 - x)**7 -
                    0.31993486318965 * (4.8 - x)**8)),
        ((2.25, 6.5),
         lambda x: (0.30601966016455 * (6.5 - x)**4 -
                    0.63945082587403 * (6.5 - x)**5 +
                    0.54057725028875 * (6.5 - x)**6 -
                    0.21210673993915 * (6.5 - x)**7 +
                    0.032014318882870 * (6.5 - x)**8))
    ]
    vals = [func(r)
            for ((lowcut, highcut), func) in funcs if lowcut <= r < highcut]
    return sum(vals)


def atsim_pairwise_alfe(r):
    """
    The pairwise potential of Al-Fe.
    """
    funcs = [
        ((0.0, 1.2),
         zerowrap(lambda x: (4867.1182946454 / x) *
                            (0.1818 * math.exp(-25.834107666296 * x) +
                             0.5099 * math.exp(-7.6073373918597 * x) +
                             0.2802 * math.exp(-3.2526756183596 * x) +
                             0.02817 * math.exp(-1.6275487829767 * x)))),
        ((1.2, 2.2),
         lambda x: math.exp(6.6167846784367 -
                            1.5208197629514 * x -
                            0.73055022396300 * x**2 -
                            0.038792724942647 * x**3)),
        ((2.2, 3.2),
         lambda x: (-4.1487019439249 * (3.2 - x)**4 +
                    5.6697481153271 * (3.2 - x)**5 -
                    1.7835153896441 * (3.2 - x)**6 -
                    3.3886912738827 * (3.2 - x)**7 +
                    1.9720627768230 * (3.2 - x)**8)),
        ((2.2, 6.2),
         lambda x: (0.094200713038410 * (6.2 - x)**4 -
                    0.16163849208165 * (6.2 - x)**5 +
                    0.10154590006100 * (6.2 - x)**6 -
                    0.027624717063181 * (6.2 - x)**7 +
                    0.0027505576632627 * (6.2 - x)**8))
    ]
    vals = [func(r)
            for ((lowcut, highcut), func) in funcs if lowcut <= r < highcut]
    return sum(vals)


def atsim_pairwise_fefe(r):
    """
    The pairwise potential of Fe-Fe.
    """
    funcs = [
        ((0.0, 1.0),
         zerowrap(lambda x: (9734.2365892908 / x) *
                            (0.1818 * math.exp(-28.616724320005 * x) +
                             0.5099 * math.exp(-8.4267310396064 * x) +
                             0.2802 * math.exp(-3.6030244464156 * x) +
                             0.02817 * math.exp(-1.8028536321603 * x)))),
        ((1.0, 2.05),
         lambda x: math.exp(7.4122709384068 -
                            0.64180690713367 * x -
                            2.6043547961722 * x**2 +
                            0.62625393931230 * x**3)),
        ((2.05, 2.2), lambda x: -27.444805994228 * (2.2 - x) ** 3),
        ((2.05, 2.3), lambda x: 15.738054058489 * (2.3 - x) ** 3),
        ((2.05, 2.4), lambda x: 2.2077118733936 * (2.4 - x) ** 3),
        ((2.05, 2.5), lambda x: -2.4989799053251 * (2.5 - x) ** 3),
        ((2.05, 2.6), lambda x: 4.2099676494795 * (2.6 - x) ** 3),
        ((2.05, 2.7), lambda x: -0.77361294129713 * (2.7 - x) ** 3),
        ((2.05, 2.8), lambda x: 0.80656414937789 * (2.8 - x) ** 3),
        ((2.05, 3.0), lambda x: -2.3194358924605 * (3.0 - x) ** 3),
        ((2.05, 3.3), lambda x: 2.6577406128280 * (3.3 - x) ** 3),
        ((2.05, 3.7), lambda x: -1.0260416933564 * (3.7 - x) ** 3),
        ((2.05, 4.2), lambda x: 0.35018615891957 * (4.2 - x) ** 3),
        ((2.05, 4.7), lambda x: -0.058531821042271 * (4.7 - x) ** 3),
        ((2.05, 5.3), lambda x: -0.0030458824556234 * (5.3 - x) ** 3)]
    vals = [func(r) for
            ((lowcut, highcut), func) in funcs if lowcut <= r < highcut]
    return sum(vals)


def test_msah11(precision=Precision.high, delta=1e-12):
    """
    Test the tensorflow based implementation of MSAH11 Al-Fe potential.
    """
    set_float_precision(precision)
    dtype = get_float_dtype()
    np_dtype = dtype.as_numpy_dtype

    rho = np.linspace(0.0, 20.0, num=201, endpoint=True)
    embed_al = np.asarray([atsim_embed_al(x) for x in rho]).astype(np_dtype)
    embed_fe = np.asarray([atsim_embed_fe(x) for x in rho]).astype(np_dtype)

    r = np.linspace(0.0, 10.0, num=101, endpoint=True)
    rho_alal = np.asarray([atsim_density_alal(x) for x in r]).astype(np_dtype)
    rho_fefe = np.asarray([atsim_density_fefe(x) for x in r]).astype(np_dtype)
    rho_alfe = np.asarray([atsim_density_alfe(x) for x in r]).astype(np_dtype)
    phi_alal = np.asarray([atsim_pairwise_alal(x) for x in r]).astype(np_dtype)
    phi_fefe = np.asarray([atsim_pairwise_fefe(x) for x in r]).astype(np_dtype)
    phi_alfe = np.asarray([atsim_pairwise_alfe(x) for x in r]).astype(np_dtype)

    with tf.Graph().as_default():

        with tf.name_scope("Inputs"):
            rho = tf.convert_to_tensor(rho, name='rho', dtype=dtype)
            r = tf.convert_to_tensor(r, name='r', dtype=dtype)

        pot = AlFeMsah11()
        embed_al_op = pot.embed(rho, element='Al', variable_scope='Embed/Al')
        embed_fe_op = pot.embed(rho, element='Fe', variable_scope='Embed/Fe')
        rho_alal_op = pot.rho(r, kbody_term='AlAl', variable_scope='Rho/AlAl')
        rho_fefe_op = pot.rho(r, kbody_term='FeFe', variable_scope='Rho/FeFe')
        rho_alfe_op = pot.rho(r, kbody_term='AlFe', variable_scope='Rho/AlFe')
        phi_alal_op = pot.phi(r, kbody_term='AlAl', variable_scope='Phi/AlAl')
        phi_fefe_op = pot.phi(r, kbody_term='FeFe', variable_scope='Phi/FeFe')
        phi_alfe_op = pot.phi(r, kbody_term='AlFe', variable_scope='Phi/AlFe')

        with tf.Session() as sess:
            embed_vals = sess.run([embed_al_op, embed_fe_op])
            rho_vals = sess.run([rho_alal_op, rho_fefe_op, rho_alfe_op])
            phi_vals = sess.run([phi_alal_op, phi_fefe_op, phi_alfe_op])

        assert_array_almost_equal(embed_vals[0], embed_al, delta=delta)
        assert_array_almost_equal(embed_vals[1], embed_fe, delta=delta)
        assert_array_almost_equal(rho_vals[0], rho_alal, delta=delta)
        assert_array_almost_equal(rho_vals[1], rho_fefe, delta=delta)
        assert_array_almost_equal(rho_vals[2], rho_alfe, delta=delta)
        assert_array_almost_equal(phi_vals[0], phi_alal, delta=delta)
        assert_array_almost_equal(phi_vals[1], phi_fefe, delta=delta)
        assert_array_almost_equal(phi_vals[2], phi_alfe, delta=delta)

    set_float_precision(Precision.high)


def test_msah11_float32():
    """
    Test the tensorflow based implementation of MSAH11 Al-Fe potential using
    float32.
    """
    test_msah11(Precision.medium, delta=1e-2)


if __name__ == "__main__":
    nose.run()
