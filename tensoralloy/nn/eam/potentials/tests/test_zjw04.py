# coding=utf-8
"""
This module defines unit tests for ZJW04 functions.

The reference codes are provided by `atsim.potentials`:

https://atsimpotentials.readthedocs.io/en/latest/index.html

"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import math
from nose.tools import assert_equal

from tensoralloy.misc import AttributeDict
from tensoralloy.test_utils import assert_array_equal
from tensoralloy.nn.eam.potentials.zjw04 import Zjw04
from tensoralloy.nn.utils import GraphKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def atsim_make_func(a, b, r_e, c):
    """
    Creates functions of the form used for density function.
    Functional form also forms components of pair potential.
    """
    def func(r):
        """
        The exponential-type potential function.
        """
        return (a * math.exp(-b * (r / r_e - 1))) / (1 + (r / r_e - c) ** 20.0)
    return func


def atsim_make_pairpot_aa(A, gamma, r_e, kappa,
                          B, omega, lamda):
    """
    Function factory that returns functions parameterised for homogeneous pair
    interactions.
    """
    f1 = atsim_make_func(A, gamma, r_e, kappa)
    f2 = atsim_make_func(B, omega, r_e, lamda)

    def func(r):
        """
        Return f1(r) - f2(r)
        """
        return f1(r) - f2(r)
    return func


def atsim_make_pairpot_ab(dens_a, phi_aa, dens_b, phi_bb):
    """
    Function factory that returns functions parameterised for heterogeneous pair
    interactions.
    """
    def func(r):
        """
        phi_ab(r) = 0.5 * (
            rho_a(r) / rho_b(r) * phi_bb(r) + rho_b(r) / rho_a(r) * phi_aa(r)
        )
        """
        v1 = dens_b(r) / dens_a(r) * phi_aa(r)
        v2 = dens_a(r) / dens_b(r) * phi_bb(r)
        return 0.5 * (v1 + v2)
    return func


def atsim_make_embed(rho_e, rho_s, F_ni, F_i, F_e, eta):
    """
    Function factory returning parameterised embedding function.
    """
    rho_n = 0.85 * rho_e
    rho_0 = 1.15 * rho_e

    def e1(rho):
        """
        rho < rho_n
        """
        return sum([F_ni[i] * (rho / rho_n - 1) ** float(i) for i in range(4)])

    def e2(rho):
        """
        rho_n <= rho < rho_0
        """
        return sum([F_i[i] * (rho / rho_e - 1) ** float(i) for i in range(4)])

    def e3(rho):
        """
        rho_0 < rho
        """
        return F_e * (1.0 - eta * math.log(rho / rho_s)) * (rho / rho_s) ** eta

    def func(rho):
        """
        Return the embedding energy.
        """
        if rho < rho_n:
            return e1(rho)
        elif rho_n <= rho < rho_0:
            return e2(rho)
        return e3(rho)
    return func


def atsim_make_functions():
    """
    Return the potential functions implemented by ATSIM.
    """
    # Potential parameters
    r_eCu = 2.556162
    f_eCu = 1.554485
    gamma_Cu = 8.127620
    omega_Cu = 4.334731
    A_Cu = 0.396620
    B_Cu = 0.548085
    kappa_Cu = 0.308782
    lambda_Cu = 0.756515

    rho_e_Cu = 21.175871
    rho_s_Cu = 21.175395
    F_ni_Cu = [-2.170269, -0.263788, 1.088878, -0.817603]
    F_i_Cu = [-2.19, 0.0, 0.561830, -2.100595]
    eta_Cu = 0.310490
    F_e_Cu = -2.186568

    r_eAl = 2.863924
    f_eAl = 1.403115
    gamma_Al = 6.613165
    omega_Al = 3.527021
    # A_Al      = 0.134873
    A_Al = 0.314873
    B_Al = 0.365551
    kappa_Al = 0.379846
    lambda_Al = 0.759692

    rho_e_Al = 20.418205
    rho_s_Al = 23.195740
    F_ni_Al = [-2.807602, -0.301435, 1.258562, -1.247604]
    F_i_Al = [-2.83, 0.0, 0.622245, -2.488244]
    eta_Al = 0.785902
    F_e_Al = -2.824528

    # Define the density functions
    dens_Cu = atsim_make_func(f_eCu, omega_Cu, r_eCu, lambda_Cu)
    dens_Al = atsim_make_func(f_eAl, omega_Al, r_eAl, lambda_Al)

    # Finally, define embedding functions for each species
    embed_Cu = atsim_make_embed(
        rho_e_Cu, rho_s_Cu, F_ni_Cu, F_i_Cu, F_e_Cu, eta_Cu)
    embed_Al = atsim_make_embed(
        rho_e_Al, rho_s_Al, F_ni_Al, F_i_Al, F_e_Al, eta_Al)

    # Define pair functions
    pair_CuCu = atsim_make_pairpot_aa(A_Cu, gamma_Cu, r_eCu, kappa_Cu,
                                      B_Cu, omega_Cu, lambda_Cu)

    pair_AlAl = atsim_make_pairpot_aa(A_Al, gamma_Al, r_eAl, kappa_Al,
                                      B_Al, omega_Al, lambda_Al)

    pair_AlCu = atsim_make_pairpot_ab(dens_Cu, pair_CuCu, dens_Al, pair_AlAl)

    functions = AttributeDict(
        rho=AttributeDict(Cu=dens_Cu, Al=dens_Al),
        embed=AttributeDict(Cu=embed_Cu, Al=embed_Al),
        phi=AttributeDict(AlAl=pair_AlAl, AlCu=pair_AlCu, CuCu=pair_CuCu)
    )
    return functions


# Initialize the ATSIM potential functions
ATSIM = atsim_make_functions()


def test_rho_phi_aa():
    """
    Test the functions `AlCuZJW04.rho` and `AlCuZJW04.phi` for r_{AA}.
    """
    r = np.linspace(0.0, 5.0, num=51, endpoint=True)
    rho_al = np.asarray([ATSIM.rho.Al(r[i]) for i in range(len(r))])
    rho_cu = np.asarray([ATSIM.rho.Cu(r[i]) for i in range(len(r))])
    phi_al = np.asarray([ATSIM.phi.AlAl(r[i]) for i in range(len(r))])
    phi_cu = np.asarray([ATSIM.phi.CuCu(r[i]) for i in range(len(r))])

    with tf.Graph().as_default():

        r = tf.convert_to_tensor(r, name='r')

        layer = Zjw04()

        with tf.name_scope("Rho"):
            with tf.name_scope("Al"):
                rho_al_op = layer.rho(r, 'Al', variable_scope='Rho/Al')
            with tf.name_scope("Cu"):
                rho_cu_op = layer.rho(r, 'Cu', variable_scope='Rho/Cu')

        with tf.name_scope("Phi"):
            with tf.name_scope("AlAl"):
                phi_al_op = layer.phi(r, 'AlAl', variable_scope='Phi/AlAl')
            with tf.name_scope("CuCu"):
                phi_cu_op = layer.phi(r, 'CuCu', variable_scope='Phi/CuCu')

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            results = sess.run([rho_al_op, rho_cu_op, phi_al_op, phi_cu_op])

        collection = tf.get_collection(GraphKeys.EAM_POTENTIAL_VARIABLES)
        assert_equal(len(collection), 16)

        assert_array_equal(results[0], rho_al)
        assert_array_equal(results[1], rho_cu)
        assert_array_equal(results[2], phi_al)
        assert_array_equal(results[3], phi_cu)


def test_phi_ab():
    """
    Test the function `AlCuZJW04.phi` for r_{AB}.
    """
    r = np.linspace(0.0, 5.0, num=51, endpoint=True)
    ref = np.asarray([ATSIM.phi.AlCu(r[i]) for i in range(len(r))])

    with tf.Graph().as_default():

        r = tf.convert_to_tensor(r, name='r')

        layer = Zjw04()

        with tf.name_scope("Phi"):
            with tf.name_scope("AlCu"):
                op = layer.phi(r, 'AlCu', variable_scope='Phi/AlCu')

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            results = sess.run(op)

        assert_array_equal(results, ref)


def test_embed():
    """
    Test the embedding function `AlCuZJW04.embed`.
    """
    rho = np.linspace(0.0, 20.0, num=201, endpoint=True)
    ref_al = np.asarray([ATSIM.embed.Al(rho[i]) for i in range(len(rho))])
    ref_cu = np.asarray([ATSIM.embed.Cu(rho[i]) for i in range(len(rho))])

    with tf.Graph().as_default():

        rho = tf.convert_to_tensor(rho, name='rho')

        layer = Zjw04()

        with tf.name_scope("Embed"):

            with tf.name_scope("Al"):
                al_op = layer.embed(rho, "Al", variable_scope='Embed/Al')
            with tf.name_scope("Cu"):
                cu_op = layer.embed(rho, "Cu", variable_scope='Embed/Cu')

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                results = sess.run([al_op, cu_op])

        assert_array_equal(results[0], ref_al)
        assert_array_equal(results[1], ref_cu)


if __name__ == "__main__":
    nose.run()
