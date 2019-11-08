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
from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.convolutional import convolution1x1

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class DeepPotSE(AtomicNN):
    """
    The tensorflow based implementation of the DeepPot-SE model.
    """

    default_collection = GraphKeys.DEEPMD_VARIABLES

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
                 atomic_static_energy=None,
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
            atomic_static_energy=atomic_static_energy,
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
        return {"class": self.__class__.__name__,
                "elements": self._elements,
                "hidden_sizes": self._hidden_sizes,
                "activation": self._activation,
                "kernel_initializer": self._kernel_initializer,
                "m1": self._m1,
                "m2": self._m2,
                "rcs": self._rcs,
                "embedding_activation": self._embedding_activation,
                "embedding_sizes": self._embedding_sizes,
                "use_resnet_dt": self._use_resnet_dt,
                'use_atomic_static_energy': self._use_atomic_static_energy,
                "minimize_properties": self._minimize_properties,
                "export_properties": self._export_properties}

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
                dx = tf.squeeze(tf.expand_dims(value[1], axis=-1),
                                axis=1, name='dx')
                dy = tf.squeeze(tf.expand_dims(value[2], axis=-1),
                                axis=1, name='dy')
                dz = tf.squeeze(tf.expand_dims(value[3], axis=-1),
                                axis=1, name='dz')
                rij = tf.squeeze(tf.expand_dims(value[0], axis=-1),
                                 axis=1, name='rij')
                sij = deepmd_cutoff(rij, self._transformer.rc, self._rcs)
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

                rr = tf.concat((sij, dx, dy, dz), axis=-1, name='R')
                d1 = tf.einsum('ijkl,ijkp->ijlp', g1, rr)
                d2 = tf.einsum('ijkl,ijpl->ijkp', d1, rr)
                d3 = tf.einsum('ijkl,ijlp->ijkp', d2, g2)
                shape = tf.shape(d3)
                x = tf.reshape(d3, [shape[0], shape[1], self._m1 * self._m2],
                               name='D')
                if verbose:
                    log_tensor(x)
                outputs[kbody_term] = x
        return self._dynamic_stitch(outputs, max_occurs, symmetric=False)

    def _dynamic_partition(self,
                           descriptors: dict,
                           mode: tf_estimator.ModeKeys,
                           merge_symmetric=True):
        """
        Split the descriptors of type `Dict[element, (tensor, mask)]` to `Np`
        partitions where `Np` is the total number of unique k-body terms.

        If `merge_symmetric` is False, `Np` is equal to `N**2`.
        If `merge_symmetric` is True, `Np` will be `N * (N + 1) / 2`.

        Here N denotes the total number of elements.

        Parameters
        ----------
        descriptors : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are elements and values are tuples of (value, mask)
            where where `value` represents the descriptors and `mask` is
            the value mask. `value` and `mask` have the same shape.
                * If `mode` is TRAIN or EVAL, both should be 4D tensors of shape
                  `[4, batch_size, max_n_terms, max_n_element, nnl]`.
                * If `mode` is PREDICT, `value` should be a 4D tensor of shape
                  `[4, max_n_terms, max_n_element, nnl]` and `mask` should be
                  a 3D tensor of shape `[max_n_terms, max_n_element, nnl]`.
            The size of the first axis is fixed to 4. Here 4 denotes:
                * r  = 0
                * dx = 1
                * dy = 2
                * dz = 3
            and `r = sqrt(dx * dx + dy * dy + dz * dz)`
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        merge_symmetric : bool
            A bool.

        Returns
        -------
        partitions : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, 1 + delta, max_n_element, nnl]`. `delta` will be zero
            if the corresponding kbody term has only one type of atom; otherwise
            `delta` will be one.
        max_occurs : Counter
            The maximum occurance of each type of element.

        """
        partitions = dict()
        max_occurs = {}
        if merge_symmetric:
            name_scope = "Partition/Symmetric"
        else:
            name_scope = "Partition"

        with tf.name_scope(name_scope):

            for element in self._elements:

                with tf.name_scope(f"{element}"):
                    kbody_terms = self._kbody_terms[element]
                    values, masks = descriptors[element]
                    values = tf.convert_to_tensor(values, name='values')
                    masks = tf.convert_to_tensor(masks, name='masks')

                    # For the ADP model, the interatomic distances `r` and
                    # differences `dx`, `dy` and `dz` are all required.
                    if mode == tf_estimator.ModeKeys.PREDICT:
                        assert values.shape.ndims == 4
                        values = tf.expand_dims(values, axis=1)
                        masks = tf.expand_dims(masks, axis=0)
                        max_occurs[element] = tf.shape(values)[3]
                    else:
                        assert values.shape.ndims == 5
                        max_occurs[element] = values.shape[3].value

                    num = len(kbody_terms)
                    glists = tf.split(
                        values, num_or_size_splits=num, axis=2, name='glist')
                    mlists = tf.split(
                        masks, num_or_size_splits=num, axis=1, name='mlist')

                    for i, (value, mask) in enumerate(zip(glists, mlists)):
                        kbody_term = kbody_terms[i]
                        if merge_symmetric:
                            kbody_term = ''.join(
                                sorted(get_elements_from_kbody_term(
                                    kbody_term)))
                            if kbody_term in partitions:
                                value = tf.concat(
                                    (partitions[kbody_term][0], value), axis=3)
                                mask = tf.concat(
                                    (partitions[kbody_term][1], mask), axis=2)
                        partitions[kbody_term] = (value, mask)
            return partitions, Counter(max_occurs)

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
            A dict of (element, (value, mask)) where `element` represents the
            symbol of an element, `value` is the descriptors of `element` and
            `mask` is the mask of `value`.
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
            partitions, max_occurs = self._dynamic_partition(
                descriptors=descriptors,
                mode=mode,
                merge_symmetric=False)

            embeddings = self._build_embedding_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                verbose=verbose)

            with tf.variable_scope("ANN"):
                activation_fn = get_activation_fn(self._activation)
                outputs = []
                for element, (_, atom_mask) in descriptors.items():
                    if self._use_atomic_static_energy:
                        bias_mean = self._atomic_static_energy.get(element, 0.0)
                    else:
                        bias_mean = 0.0
                    with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                        x = embeddings[element]
                        if verbose:
                            log_tensor(x)
                        hidden_sizes = self._hidden_sizes[element]
                        yi = convolution1x1(
                            x,
                            activation_fn=activation_fn,
                            hidden_sizes=hidden_sizes,
                            num_out=1,
                            l2_weight=1.0,
                            collections=collections,
                            output_bias=self._use_atomic_static_energy,
                            output_bias_mean=bias_mean,
                            use_resnet_dt=self._use_resnet_dt,
                            kernel_initializer="he_normal",
                            variable_scope=None,
                            verbose=verbose)
                        yi = tf.squeeze(yi, axis=2, name='atomic')
                        if verbose:
                            log_tensor(yi)
                        outputs.append(yi)
                return outputs
