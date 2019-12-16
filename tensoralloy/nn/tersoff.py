# coding=utf-8
"""
The Tersoff potential.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator
from typing import List, Dict
from collections import Counter

from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.descriptor.cutoff import tersoff_cutoff
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.utils import log_tensor
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

    def _dynamic_stitch(self,
                        outputs: Dict[str, tf.Tensor],
                        max_occurs: Counter,
                        symmetric=False):
        """
        The reverse of `dynamic_partition`. Interleave the kbody-term centered
        `outputs` of type `Dict[kbody_term, tensor]` to element centered values
        of type `Dict[element, tensor]`.

        Parameters
        ----------
        outputs : Dict[str, tf.Tensor]
            A dict. The keys are unique kbody-terms and values are 2D tensors
            with shape `[batch_size, max_n_elements]` where `max_n_elements`
            denotes the maximum occurance of the center element of the
            corresponding kbody-term.
        max_occurs : Counter
            The maximum occurance of each type of element.
        symmetric : bool
            This should be True if kbody terms all symmetric.

        Returns
        -------
        results : Dict
            A dict. The keys are elements and the values are corresponding
            merged features.

        """
        with tf.name_scope("Stitch"):
            stacks: Dict = {}
            for kbody_term, value in outputs.items():
                center, other = get_elements_from_kbody_term(kbody_term)
                if symmetric and center != other:
                    sizes = [max_occurs[center], max_occurs[other]]
                    splits = tf.split(
                        value, sizes, axis=1, name=f'splits/{center}{other}')
                    stacks[center] = stacks.get(center, []) + [splits[0]]
                    stacks[other] = stacks.get(other, []) + [splits[1]]
                else:
                    stacks[center] = stacks.get(center, []) + [value]
            return tf.concat(
                [tf.add_n(stacks[el], name=el) for el in self._elements],
                axis=1, name='sum')

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
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
        descriptors : Dict
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
                one = tf.convert_to_tensor(1.0, dtype=dtype, name='one')
                half = tf.convert_to_tensor(0.5, dtype=dtype, name='half')

            zeta = {}

            with tf.name_scope("Angular"):
                angular = dynamic_partition(
                    dists_and_masks=descriptors["angular"],
                    elements=clf.elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    merge_symmetric=False,
                    angular=True)[0]

                for kbody_term, (dists, masks) in angular.items():
                    center, mid = get_elements_from_kbody_term(kbody_term)[0: 2]
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
                        upper = tf.math.subtract(rij2 + rik2, rjk2, name='upper')
                        lower = tf.math.multiply(
                            tf.math.multiply(two, rij), rik, name='lower')
                        costheta = tf.math.divide_no_nan(upper, lower, name='costheta')
                        gtheta = tf.math.multiply(
                            gamma,
                            tf.math.add(one, c2 / d2 - c2 / (d2 + (costheta - costheta0) ** 2)),
                            name='gtheta')
                        fc = tersoff_cutoff(rik, R, D, name='fc')
                        l3m = safe_pow(lambda3, m)
                        drijk = tf.math.subtract(rij, rik)
                        drijkm = safe_pow(drijk, m)
                        z = tf.math.multiply(fc * gtheta,
                                             tf.math.exp(l3m * drijkm),
                                             name='zeta/single')
                        z = tf.reduce_sum(z, axis=-1, keepdims=True, name='zeta')
                        zeta[f'{center}{mid}'] = z

            outputs = {}

            with tf.name_scope("Radial"):
                radial, max_occurs = dynamic_partition(
                    dists_and_masks=descriptors["radial"],
                    elements=self._elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=False,
                    merge_symmetric=False)
                for kbody_term, (dists, masks) in radial.items():
                    center, pair = get_elements_from_kbody_term(kbody_term)
                    with tf.name_scope("Parameters"):
                        R = tf.convert_to_tensor(3.0, dtype=dtype, name='R')
                        D = tf.convert_to_tensor(0.2, dtype=dtype, name='D')
                        A = tf.convert_to_tensor(3264.7, dtype=dtype, name='A')
                        B = tf.convert_to_tensor(0.2, dtype=dtype, name='B')
                        lambda1 = tf.convert_to_tensor(3.2394, dtype=dtype, name='lambda3')
                        lambda2 = tf.convert_to_tensor(95.373, dtype=dtype, name='lambda3')
                        beta = tf.convert_to_tensor(0.33675, dtype=dtype, name='beta')
                        n = tf.convert_to_tensor(22.956, dtype=dtype, name='n')
                    with tf.variable_scope(f"{kbody_term}"):
                        rij = tf.squeeze(dists[0], axis=1, name='rij')
                        fc = tersoff_cutoff(rij, R, D, name='fc')
                        fr = tf.math.multiply(A, tf.exp(tf.negative(lambda1 * rij)), name='fr')
                        fa = tf.negative(tf.math.multiply(B, tf.exp(tf.negative(lambda2 * rij))), name='fa')
                        bz = tf.math.multiply(beta, zeta[f'{center}{pair}'], name='bz')
                        bzn = safe_pow(bz, n)
                        coef = tf.math.truediv(one, -two * n, name='2ni')
                        bij = safe_pow(one + bzn, coef)
                        vij = tf.math.multiply(fc, fr + bij * fa, name='vij')
                        vij = tf.math.multiply(half, vij, name='vij/half')
                        outputs[kbody_term] = tf.reduce_sum(vij, axis=[-1, -2], keepdims=False, name='vij/sum')

            y_atomic = self._dynamic_stitch(outputs, max_occurs)
            if mode == tf_estimator.ModeKeys.PREDICT:
                y_atomic = tf.squeeze(y_atomic, name='atomic/squeeze')
            return y_atomic

    def _get_internal_energy_op(self, outputs: tf.Tensor, features: dict,
                                name='energy', verbose=True):
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.
        features : Dict
            A dict of input features.
        name : str
            The name of the output tensor.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        y_atomic = tf.identity(outputs, name='y_atomic')
        ndims = features["atom_masks"].shape.ndims
        axis = ndims - 1
        with tf.name_scope("Mask"):
            mask = tf.split(
                features["atom_masks"], [1, -1], axis=axis, name='split')[1]
            y_mask = tf.multiply(y_atomic, mask, name='mask')
            self._y_atomic_op_name = y_mask.name
        energy = tf.reduce_sum(
            y_mask, axis=axis, keepdims=False, name=name)
        if verbose:
            log_tensor(energy)
        return energy
