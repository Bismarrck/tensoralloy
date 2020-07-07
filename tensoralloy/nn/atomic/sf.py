#!coding=utf-8
"""
The symmetry function based atomistic neural network.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from tensorflow_estimator import estimator as tf_estimator
from typing import List
from sklearn.model_selection import ParameterGrid

from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.atomic import AtomicNN
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors
from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.extension.grad_ops import safe_pow
from tensoralloy.precision import get_float_dtype
from tensoralloy.descriptor.cutoff import cosine_cutoff, polynomial_cutoff

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class SymmetryFunctionNN(AtomicNN):
    """
    Symmetry function based atomistic neural network potential.
    """

    def __init__(self,
                 elements: List[str],
                 hidden_sizes=None,
                 activation=None,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces'),
                 kernel_initializer="he_normal",
                 minmax_scale=False,
                 use_atomic_static_energy=True,
                 fixed_atomic_static_energy=False,
                 atomic_static_energy=None,
                 use_resnet_dt=True,
                 temperature_dependent=False,
                 temperature_layers=(128, 128),
                 temperature_activation='softplus',
                 eta=np.array([0.05, 4.0, 20.0, 80.0]),
                 omega=np.asarray([0.0]),
                 beta=np.asarray([0.005]),
                 gamma=np.asarray([1.0, -1.0]),
                 zeta=np.asarray([1.0, 4.0]),
                 cutoff_function="cosine"):
        super(SymmetryFunctionNN, self).__init__(
            elements=elements,
            hidden_sizes=hidden_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_atomic_static_energy=use_atomic_static_energy,
            fixed_atomic_static_energy=fixed_atomic_static_energy,
            atomic_static_energy=atomic_static_energy,
            use_resnet_dt=use_resnet_dt,
            minmax_scale=minmax_scale,
            temperature_dependent=temperature_dependent,
            temperature_layers=temperature_layers,
            temperature_activation=temperature_activation,
            minimize_properties=minimize_properties,
            export_properties=export_properties)

        self._eta = eta
        self._omega = omega
        self._gamma = gamma
        self._zeta = zeta
        self._beta = beta
        self._cutoff_function = cutoff_function
        self._nn_scope = "Atomic/SF"
        self._radial_parameters = ParameterGrid({'eta': self._eta,
                                                 'omega': self._omega})
        self._angular_parameters = ParameterGrid({'beta': self._beta,
                                                  'zeta': self._zeta,
                                                  'gamma': self._gamma})

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        d = super(SymmetryFunctionNN, self).as_dict()
        d["eta"] = self._eta
        d["omega"] = self._omega
        d["gamma"] = self._gamma
        d["zeta"] = self._zeta
        d["beta"] = self._beta
        d["cutoff_function"] = self._cutoff_function
        return d

    def _apply_cutoff(self, x, rc, name=None):
        """
        Apply the cutoff function on interatomic distances.
        """
        if self._cutoff_function == "cosine":
            return cosine_cutoff(x, rc, name=name)
        else:
            return polynomial_cutoff(x, rc, name=name)

    def _apply_g2_functions(self, partitions: dict):
        """
        Apply the G2 symmetry functions on the partitions.
        """
        clf = self._transformer
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

    def _apply_g4_functions(self, partitions: dict):
        """
        Apply the G4 symmetry functions on the partitions.
        """
        clf = self._transformer
        assert isinstance(clf, UniversalTransformer)
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

    def get_symmetry_function_descriptors(self,
                                          universal_descriptors: dict,
                                          mode: tf_estimator.ModeKeys):
        """
        Construct the computation graph for calculating symmetry function
        descriptors.
        """
        clf = self._transformer
        assert isinstance(clf, UniversalTransformer)
        with tf.name_scope("Radial"):
            partitions, max_occurs = dynamic_partition(
                dists_and_masks=universal_descriptors['radial'],
                elements=clf.elements,
                kbody_terms_for_element=clf.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=False)
            g = self._apply_g2_functions(partitions)
        if clf.angular:
            with tf.name_scope("Angular"):
                partitions = dynamic_partition(
                    dists_and_masks=universal_descriptors['angular'],
                    elements=clf.elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=True,
                    merge_symmetric=False)[0]
                g4 = self._apply_g4_functions(partitions)

            for e, tensor in g4.items():
                g[e] = tf.concat((g[e], tensor), axis=-1)
        return g, max_occurs

    def _get_atomic_descriptors(self,
                                universal_descriptors,
                                mode: tf_estimator.ModeKeys,
                                verbose=True):
        """
        A wrapper.
        """
        descriptors, max_occurs = self.get_symmetry_function_descriptors(
            universal_descriptors, mode)
        return AtomicDescriptors(descriptors=descriptors, max_occurs=max_occurs)


class TemperatureDependentSymmetryFunctionNN(SymmetryFunctionNN):
    """
    Temperature-dependent symmetry function atomistic neural network potential.
    """

    def __init__(self, *args, **kwargs):
        """ Initialization method. """
        super(TemperatureDependentSymmetryFunctionNN, self).__init__(
            *args, **kwargs)
        self._nn_scope = "Atomic/TDSF"

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
                           mode: tf_estimator.ModeKeys,
                           verbose=False):
        """
        Return the temperature-dependent model outputs.

        Parameters
        ----------
        features : Dict
            A dict of tensors, includeing raw properties and the descriptors:
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        descriptors : Dict
            A dict of (element, (dists, masks)) as the universal descriptors.
            Here `element` represents the symbol of an element.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 1D (PREDICT) or 2D (TRAIN or EVAL) tensor as the unmasked atomic
            energies of atoms. The last axis has the size `max_n_atoms`.

        """
        with tf.variable_scope(self._nn_scope, reuse=tf.AUTO_REUSE):
            collections = [self.default_collection]

            with tf.variable_scope("ANN"):
                activation_fn = get_activation_fn(self._activation)
                outputs = []
                symmetry_function_descriptors, max_occurs = \
                    self.get_symmetry_function_descriptors(descriptors, mode)
                for element, x in symmetry_function_descriptors.items():
                    if self._use_atomic_static_energy:
                        bias_mean = self._atomic_static_energy.get(element, 0.0)
                    else:
                        bias_mean = 0.0
                    with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                        if self._minmax_scale:
                            x = self._apply_minmax_normalization(
                                x=x,
                                mask=descriptors['atom_masks'][element],
                                mode=mode,
                                collections=collections)
                        if verbose:
                            log_tensor(x)
                        hidden_sizes = self._hidden_sizes[element]
                        with tf.name_scope("etemp"):
                            if mode == tf_estimator.ModeKeys.PREDICT:
                                d0 = 1
                                d1 = max_occurs[element]
                            else:
                                d0, d1 = x.shape.as_list()[0: 2]
                            etemp = tf.reshape(
                                features['etemperature'], [1, d0],
                                name='etemp/0')
                            etemp = tf.transpose(etemp, name='T')
                            etemp = tf.tile(etemp, [d1, 1], name='tiled')
                            etemp = tf.reshape(
                                etemp, [d0, d1, 1], name='etemp')
                            x = tf.concat((x, etemp), axis=2, name='x')
                        yi = convolution1x1(
                            x,
                            activation_fn=activation_fn,
                            hidden_sizes=hidden_sizes,
                            num_out=1,
                            l2_weight=1.0,
                            collections=collections,
                            output_bias=self._use_atomic_static_energy,
                            output_bias_mean=bias_mean,
                            fixed_output_bias=self._fixed_atomci_static_energy,
                            use_resnet_dt=self._use_resnet_dt,
                            kernel_initializer="he_normal",
                            variable_scope=None,
                            verbose=verbose)
                        yi = tf.squeeze(yi, axis=2, name='atomic')
                        if verbose:
                            log_tensor(yi)
                        outputs.append(yi)
                return outputs
