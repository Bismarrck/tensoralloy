#!coding=utf-8
"""
The End-to-End symmetry preserved DeepPotSE model.
"""
from __future__ import print_function, absolute_import
from os import name

import tensorflow as tf

from typing import List, Dict
from collections import Counter

from tensoralloy.transformer import UniversalTransformer
from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms
from tensoralloy.utils import GraphKeys, ModeKeys
from tensoralloy.nn.cutoff import deepmd_cutoff
from tensoralloy.nn.utils import log_tensor, get_activation_fn
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.atomic.atomic import Descriptor
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class DeepPotSE(Descriptor):
    """
    The tensorflow based implementation of the DeepPot-SE model.
    """

    default_collection = GraphKeys.DEEPMD_VARIABLES

    def __init__(self,
                 elements: List[str],
                 rcs=5.0,
                 m1=100,
                 m2=4,
                 embedding_activation='tanh',
                 embedding_sizes=(20, 40, 80)):
        """
        Initialization method.
        """
        super(DeepPotSE, self).__init__(elements=elements)

        self._m1 = m1
        self._m2 = m2
        self._rcs = rcs
        self._embedding_activation = embedding_activation
        self._embedding_sizes = embedding_sizes
        self._kbody_terms = get_kbody_terms(self._elements, angular=False)[1]

    @property
    def name(self):
        """ Return the name of this descriptor. """
        return "DPMD"

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
                            transformer: UniversalTransformer,
                            partitions: dict,
                            max_occurs: Counter,
                            verbose=False):
        """
        Return the outputs of the embedding networks.

        Parameters
        ----------
        transformer : UniversalTransformer
            The attached universal transformer.
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
        for kbody_term, (value, masks) in partitions.items():
            with tf.variable_scope(f"{kbody_term}"):
                ijx = tf.squeeze(value[1], axis=1, name='ijx')
                ijy = tf.squeeze(value[2], axis=1, name='ijy')
                ijz = tf.squeeze(value[3], axis=1, name='ijz')
                rij = tf.squeeze(value[0], axis=1, name='rij')
                masks = tf.squeeze(masks, axis=1, name='masks')
                sij = deepmd_cutoff(rij, transformer.rc, self._rcs,
                                    name="sij")
                z = tf.div_no_nan(sij, rij, name="z")
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
                g1 = tf.multiply(g1, masks, name="G1")
                g2 = tf.identity(g1[..., :self._m2], name='G2')
                ijx = tf.multiply(ijx, z, name="ijx/hat")
                ijy = tf.multiply(ijy, z, name="ijy/hat")
                ijz = tf.multiply(ijz, z, name="ijz/hat")
                rr = tf.concat((sij, ijx, ijy, ijz), axis=-1, name='R')
                d1 = tf.einsum('ijkl,ijkp->ijlp', g1, rr, name='GR')
                d2 = tf.einsum('ijkl,ijpl->ijkp', d1, rr, name='GRR')
                d3 = tf.einsum('ijkl,ijlp->ijkp', d2, g2, name='GRRG')
                shape = tf.shape(d3)
                x = tf.reshape(d3, [shape[0], shape[1], self._m1 * self._m2],
                               name='D')
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

    def calculate(self,
                  transformer: UniversalTransformer,
                  universal_descriptors,
                  mode: ModeKeys,
                  verbose=False) -> AtomicDescriptors:
        """
        A wrapper function.
        """
        partitions, max_occurs = dynamic_partition(
                dists_and_masks=universal_descriptors['radial'],
                elements=transformer.elements,
                kbody_terms_for_element=transformer.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=False)
        embeddings = self._build_embedding_nn(
            transformer=transformer,
            partitions=partitions,
            max_occurs=max_occurs,
            verbose=verbose)
        return AtomicDescriptors(embeddings, max_occurs)
