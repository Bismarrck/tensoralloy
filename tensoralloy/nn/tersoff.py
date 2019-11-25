#coding=utf-8
"""
The Tersoff potential.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator
from typing import List

from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.descriptor.cutoff import tersoff_cutoff
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.precision import get_float_dtype
from tensoralloy.extension.grad_ops import safe_pow

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'



class Tersoff(BasicNN):
    """
    The Tersoff interaction potential.
    """

    def __init__(self,
                 elements: List[str],
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces', 'stress')):
        super(Tersoff, self).__init__(elements=elements, hidden_sizes=None,
                                      minimize_properties=minimize_properties,
                                      export_properties=export_properties)
        self._nn_scope = "Tersoff"

    def _get_model_outputs(self,
                           features: dict,
                           dists_and_masks: dict,
                           mode: tf_estimator.ModeKeys,
                           verbose=False):
        """
        Return raw NN-EAM model outputs.

        Parameters
        ----------
        features : Dict
            A dict of tensors, includeing raw properties and the descriptors:
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        dists_and_masks : Dict
            A dict of (element, (dists, masks)) where `element` represents the
            symbol of an element.
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
            clf = self._transformer
            dtype = get_float_dtype()
            assert isinstance(clf, UniversalTransformer)

            with tf.name_scope("Constants"):
                two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')

            with tf.name_scope("Radial"):
                radial = dynamic_partition(
                    dists_and_masks=dists_and_masks["radial"],
                    elements=self._elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=False,
                    merge_symmetric=False)[0]
                for kbody_term, (dists, masks) in radial.items():
                    center = get_elements_from_kbody_term(kbody_term)[0]
                    with tf.name_scope("Parameters"):
                        R = tf.convert_to_tensor(3.0, dtype=dtype, name='R')
                        D = tf.convert_to_tensor(0.2, dtype=dtype, name='D')
                        A = tf.convert_to_tensor(3264.7, dtype=dtype, name='A')
                        B = tf.convert_to_tensor(0.2, dtype=dtype, name='B')
                        lambda1 = tf.convert_to_tensor(3.2394, dtype=dtype, name='lambda3')
                        lambda2 = tf.convert_to_tensor(95.373, dtype=dtype, name='lambda3')
                    with tf.variable_scope(f"{kbody_term}"):
                        rij = tf.squeeze(dists[0], axis=1, name='rij')
                        fc = tersoff_cutoff(rij, R, D, name='fc')
                        fr = tf.math.multiply(A, tf.exp(tf.negative(lambda1 * rij)), name='fr')
                        fa = tf.negative(tf.math.multiply(B, tf.exp(tf.negative(lambda2 * rij))), name='fa')

            with tf.name_scope("Angular"):
                angular = dynamic_partition(
                    dists_and_masks=dists_and_masks["angular"],
                    elements=clf.elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=True)[0]

                for kbody_term, (dists, masks) in angular.items():
                    center = get_elements_from_kbody_term(kbody_term)[0]
                    with tf.name_scope("Parameters"):
                        gamma = tf.convert_to_tensor(1.0, dtype=dtype, name='gamma')
                        c = tf.convert_to_tensor(4.8381, dtype=dtype, name='c')
                        d = tf.convert_to_tensor(2.0417, dtype=dtype, name='d')
                        costheta0 = tf.convert_to_tensor(0.0, dtype=dtype, name='costheta0')
                        R = tf.convert_to_tensor(3.0, dtype=dtype, name='R')
                        D = tf.convert_to_tensor(0.2, dtype=dtype, name='D')
                        m = tf.convert_to_tensor(3.0, dtype=dtype, name='m')
                        lambda3 = tf.convert_to_tensor(1.3258, dtype=dtype, name='lambda3')
                    c2 = tf.square(c, name='c2')
                    d2 = tf.square(d, name='d2')
                    with tf.variable_scope(f"{kbody_term}"):
                        rij = tf.squeeze(dists[0], axis=1, name='rij')
                        rik = tf.squeeze(dists[4], axis=1, name='rik')
                        rjk = tf.squeeze(dists[8], axis=1, name='rjk')
                        rij2 = tf.square(rij, name='rij2')
                        rik2 = tf.square(rik, name='rik2')
                        rjk2 = tf.square(rjk, name='rjk2')
                        r2 = tf.add_n([rij2, rik2, rjk2], name='r2')
                        upper = tf.math.subtract(rij2 + rik2, rjk2, name='upper')
                        lower = tf.math.multiply(
                            tf.math.multiply(two, rij), rik, name='lower')
                        costheta = tf.math.divide_no_nan(upper, lower, name='costheta')
                        gtheta = tf.math.multiply(
                            gamma,
                            tf.math.add(c2 / d2 - c2 / (d2 + (costheta - costheta0)**2)),
                            name='gtheta')
                        fc = tersoff_cutoff(rik, R, D, name='fc')
                        l3m = safe_pow(lambda3, m)
                        drijk = tf.math.subtract(rij, rik)
                        drijkm = safe_pow(drijk, m)
                        zeta = tf.math.multiply(fc * gtheta,
                                                tf.math.exp(l3m * drijkm),
                                                name='zeta/single')
