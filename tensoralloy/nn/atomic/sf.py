#!coding=utf-8
"""
The symmetry function based atomistic neural network.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from typing import List
from sklearn.model_selection import ParameterGrid

from tensoralloy.transformer import UniversalTransformer
from tensoralloy.utils import get_elements_from_kbody_term, ModeKeys
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.atomic.atomic import Descriptor
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors
from tensoralloy.extension.grad_ops import safe_pow
from tensoralloy.precision import get_float_dtype
from tensoralloy.nn.cutoff import cosine_cutoff, polynomial_cutoff

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class SymmetryFunction(Descriptor):
    """
    Symmetry function based atomistic neural network potential.
    """

    def __init__(self,
                 elements: List[str],
                 eta=np.array([0.05, 4.0, 20.0, 80.0]),
                 omega=np.asarray([0.0]),
                 beta=np.asarray([0.005]),
                 gamma=np.asarray([1.0, -1.0]),
                 zeta=np.asarray([1.0, 4.0]),
                 cutoff_function="cosine"):
        super(SymmetryFunction, self).__init__(elements=elements)

        self._eta = np.asarray(eta)
        self._omega = np.asarray(omega)
        self._gamma = np.asarray(gamma)
        self._zeta = np.asarray(zeta)
        self._beta = np.asarray(beta)
        self._cutoff_function = cutoff_function
        self._radial_parameters = ParameterGrid({'eta': self._eta,
                                                 'omega': self._omega})
        self._angular_parameters = ParameterGrid({'beta': self._beta,
                                                  'zeta': self._zeta,
                                                  'gamma': self._gamma})

    @property
    def name(self):
        """ Return the name of this descriptor. """
        return "SF"

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        d = super(SymmetryFunction, self).as_dict()
        d.update({
            "eta": self._eta.tolist(), "omega": self._omega.tolist(),
            "gamma": self._gamma.tolist(), "zeta": self._zeta.tolist(),
            "beta": self._beta.tolist(),
            "cutoff_function": self._cutoff_function})
        return d

    def _apply_cutoff(self, x, rc, name=None):
        """
        Apply the cutoff function on interatomic distances.
        """
        if self._cutoff_function == "cosine":
            return cosine_cutoff(x, rc, name=name)
        else:
            return polynomial_cutoff(x, rc, name=name)

    def _apply_g2_functions(self, clf: UniversalTransformer, partitions: dict):
        """
        Apply the G2 symmetry functions on the partitions.
        """
        dtype = get_float_dtype()
        rc = tf.convert_to_tensor(clf.rcut, name='rc', dtype=dtype)
        rc2 = tf.convert_to_tensor(clf.rcut**2, dtype=dtype, name='rc2')
        outputs = {element: [None] * len(self._elements)
                   for element in self._elements}
        for kbody_term, (dists, masks) in partitions.items():
            center = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"{kbody_term}"):
                rij = tf.squeeze(dists[0], axis=1, name='rij')
                masks = tf.squeeze(masks, axis=1, name='masks')
                fc = self._apply_cutoff(rij, rc=rc, name='fc')
                gtau = []
                for tau, grid in enumerate(self._radial_parameters):
                    with tf.name_scope(f"G2/{tau}"):
                        omega = tf.convert_to_tensor(
                            grid['omega'], dtype=dtype, name='omega')
                        eta = tf.convert_to_tensor(
                            grid['eta'], dtype=dtype, name='eta')
                        z = tf.math.truediv(
                            tf.square(tf.math.subtract(rij, omega)),
                            rc2, name='z')
                        v = tf.exp(tf.negative(tf.math.multiply(z, eta)))
                        v = tf.math.multiply(v, fc)
                        v = tf.math.multiply(v, masks, name='v/masked')
                        g = tf.expand_dims(
                            tf.reduce_sum(v, axis=[-1, -2], keep_dims=False),
                            axis=-1, name='g')
                        gtau.append(g)
                g = tf.concat(gtau, axis=-1, name='g')
            index = clf.kbody_terms_for_element[center].index(kbody_term)
            outputs[center][index] = g
        with tf.name_scope("Concat"):
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    outputs[element], axis=-1, name=element)
            return results

    def _apply_g4_functions(self, clf: UniversalTransformer, partitions: dict):
        """
        Apply the G4 symmetry functions on the partitions.
        """
        dtype = get_float_dtype()
        rc = tf.convert_to_tensor(clf.acut, name='rc', dtype=dtype)
        rc2 = tf.convert_to_tensor(clf.acut**2, dtype=dtype, name='rc2')
        two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')
        one = tf.convert_to_tensor(1.0, dtype=dtype, name='one')
        n = len(self._elements)
        outputs = {element: [None] * ((n + 1) * n // 2)
                   for element in self._elements}
        for kbody_term, (dists, masks) in partitions.items():
            center = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"{kbody_term}"):
                rij = tf.squeeze(dists[0], axis=1, name='rij')
                rik = tf.squeeze(dists[4], axis=1, name='rik')
                rjk = tf.squeeze(dists[8], axis=1, name='rjk')
                masks = tf.squeeze(masks, axis=1, name='masks')
                rij2 = tf.square(rij, name='rij2')
                rik2 = tf.square(rik, name='rik2')
                rjk2 = tf.square(rjk, name='rjk2')
                r2 = tf.add_n([rij2, rik2, rjk2], name='r2')
                z = tf.math.truediv(r2, rc2, name='z')
                upper = tf.math.subtract(rij2 + rik2, rjk2, name='upper')
                lower = tf.math.multiply(
                    tf.math.multiply(two, rij), rik, name='lower')
                theta = tf.math.divide_no_nan(upper, lower, name='theta')
                fc_rij = self._apply_cutoff(rij, rc=rc, name='fc/rij')
                fc_rik = self._apply_cutoff(rik, rc=rc, name='fc/rik')
                fc_rjk = self._apply_cutoff(rjk, rc=rc, name='fc/rjk')
                fc = tf.multiply(fc_rij, fc_rik * fc_rjk, name='fc')
                gtau = []
                for tau, grid in enumerate(self._angular_parameters):
                    with tf.name_scope(f"G4/{tau}"):
                        beta = tf.convert_to_tensor(
                            grid['beta'], dtype=dtype, name='beta')
                        zeta = tf.convert_to_tensor(
                            grid['zeta'], dtype=dtype, name='zeta')
                        gamma = tf.convert_to_tensor(
                            grid['gamma'], dtype=dtype, name='gamma')
                        outer = safe_pow(two, one - zeta)
                        v = tf.math.multiply(
                            safe_pow(one + gamma * theta, zeta),
                            tf.math.multiply(
                                tf.exp(tf.negative(tf.math.multiply(z, beta))),
                                fc))
                        v = tf.math.multiply(v, outer)
                        v = tf.math.multiply(v, masks, name='v/masked')
                        g = tf.expand_dims(
                            tf.reduce_sum(v, axis=[-1, -2], keep_dims=False),
                            axis=-1, name='g')
                        gtau.append(g)
                g = tf.concat(gtau, axis=-1, name='g')
                index = clf.kbody_terms_for_element[center].index(kbody_term)
                outputs[center][index - n] = g
        with tf.name_scope("Concat"):
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    outputs[element], axis=-1, name=element)
            return results

    def calculate(self,
                  transformer: UniversalTransformer,
                  universal_descriptors,
                  mode: ModeKeys,
                  verbose=False):
        """
        Construct the computation graph for calculating symmetry function
        descriptors.
        """
        with tf.name_scope("Radial"):
            partitions, max_occurs = dynamic_partition(
                dists_and_masks=universal_descriptors['radial'],
                elements=transformer.elements,
                kbody_terms_for_element=transformer.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=False)
            g = self._apply_g2_functions(transformer, partitions)
        if transformer.angular:
            with tf.name_scope("Angular"):
                partitions = dynamic_partition(
                    dists_and_masks=universal_descriptors['angular'],
                    elements=transformer.elements,
                    kbody_terms_for_element=transformer.kbody_terms_for_element,
                    mode=mode,
                    angular=True,
                    merge_symmetric=False)[0]
                g4 = self._apply_g4_functions(transformer, partitions)

            for e, tensor in g4.items():
                g[e] = tf.concat((g[e], tensor), axis=-1)
        return AtomicDescriptors(descriptors=g, max_occurs=max_occurs)
