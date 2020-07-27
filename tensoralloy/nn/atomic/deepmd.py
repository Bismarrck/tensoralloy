#!coding=utf-8
"""
The End-to-End symmetry preserved DeepPotSE model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Dict
from collections import Counter
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.descriptor.cutoff import deepmd_cutoff
from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms
from tensoralloy.utils import GraphKeys
from tensoralloy.nn.utils import log_tensor, get_activation_fn
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors
from tensoralloy.nn.atomic.dataclasses import FiniteTemperatureOptions

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class DeepPotSE(AtomicNN):
    """
    The tensorflow based implementation of the DeepPot-SE model.
    """

    default_collection = GraphKeys.DEEPMD_VARIABLES
    scope = "DPMD"

    def __init__(self,
                 elements: List[str],
                 rcs: float,
                 m1=100,
                 m2=4,
                 hidden_sizes=None,
                 activation=None,
                 kernel_initializer='he_normal',
                 embedding_activation='tanh',
                 embedding_sizes=(20, 40, 80),
                 use_resnet_dt=False,
                 use_atomic_static_energy=True,
                 fixed_atomic_static_energy=False,
                 atomic_static_energy=None,
                 finite_temperature=FiniteTemperatureOptions(),
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces', 'hessian')):
        """
        Initialization method.
        """
        self._nn_scope = "DeepPotSE"

        super(DeepPotSE, self).__init__(
            elements=elements,
            hidden_sizes=hidden_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            minmax_scale=False,
            use_resnet_dt=use_resnet_dt,
            use_atomic_static_energy=use_atomic_static_energy,
            fixed_atomic_static_energy=fixed_atomic_static_energy,
            atomic_static_energy=atomic_static_energy,
            finite_temperature=finite_temperature,
            minimize_properties=minimize_properties,
            export_properties=export_properties)

        self._m1 = m1
        self._m2 = m2
        self._rcs = rcs
        self._embedding_activation = embedding_activation
        self._embedding_sizes = embedding_sizes
        self._kbody_terms = get_kbody_terms(self._elements, angular=False)[1]

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        d = super(DeepPotSE, self).as_dict()
        d.update({"m1": self._m1,
                  "m2": self._m2,
                  "rcs": self._rcs,
                  "embedding_activation": self._embedding_activation,
                  "embedding_sizes": self._embedding_sizes})
        return d

    def _build_embedding_nn(self,
                            partitions: dict,
                            max_occurs: Counter,
                            verbose=False):
        """
        Return the outputs of the embedding networks.

        Parameters
        ----------
        partitions : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, 1, max_n_element, nnl]`.
        max_occurs : Counter
            The maximum occurance of each type of element.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        results : Dict
            A dict.

        """
        collections = [self.default_collection]
        outputs = {}
        for kbody_term, (value, mask) in partitions.items():
            with tf.variable_scope(f"{kbody_term}"):
                ijx = tf.squeeze(value[1], axis=1, name='ijx')
                ijy = tf.squeeze(value[2], axis=1, name='ijy')
                ijz = tf.squeeze(value[3], axis=1, name='ijz')
                rij = tf.squeeze(value[0], axis=1, name='rij')
                sij = deepmd_cutoff(rij, self._transformer.rc, self._rcs,
                                    name="sij")
                if verbose:
                    log_tensor(sij)

                activation_fn = get_activation_fn(self._embedding_activation)
                g1 = convolution1x1(sij,
                                    activation_fn=activation_fn,
                                    hidden_sizes=self._embedding_sizes,
                                    num_out=self._m1,
                                    collections=collections,
                                    # `resnet_dt` and `output_bias` are fixed to
                                    # False for embedding networks
                                    use_resnet_dt=False,
                                    output_bias=False,
                                    variable_scope=None,
                                    verbose=verbose)
                g2 = tf.identity(g1[..., :self._m2], name='g2')

                rr = tf.concat((sij, ijx, ijy, ijz), axis=-1, name='R')
                d1 = tf.einsum('ijkl,ijkp->ijlp', g1, rr, name='GR')
                d2 = tf.einsum('ijkl,ijpl->ijkp', d1, rr, name='GRR')
                d3 = tf.einsum('ijkl,ijlp->ijkp', d2, g2, name='GRRG')
                shape = tf.shape(d3)
                x = tf.reshape(d3, [shape[0], shape[1], self._m1 * self._m2],
                               name='D')
                # TODO: apply value mask on `x`
                if verbose:
                    log_tensor(x)
                outputs[kbody_term] = x
        return self._dynamic_stitch(outputs, max_occurs, symmetric=False)

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
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    stacks[element], axis=-1, name=element)
            return results

    def _get_atomic_descriptors(self,
                                universal_descriptors,
                                mode: tf_estimator.ModeKeys,
                                verbose=True):
        """
        A wrapper function.
        """
        clf = self.transformer
        partitions, max_occurs = dynamic_partition(
                dists_and_masks=universal_descriptors['radial'],
                elements=clf.elements,
                kbody_terms_for_element=clf.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=False)
        embeddings = self._build_embedding_nn(
            partitions=partitions,
            max_occurs=max_occurs,
            verbose=verbose)
        return AtomicDescriptors(embeddings, max_occurs)
