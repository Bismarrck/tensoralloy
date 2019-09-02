#!coding=utf-8
"""
This module defines the MeamNN.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator
from typing import List

from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.utils import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class MeamNN(EamAlloyNN):
    """
    The tensorflow-based implementation of the Modified Embedded-Atom Method.
    """

    def __init__(self,
                 elements: List[str],
                 custom_potentials=None,
                 hidden_sizes=None,
                 activation=None,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces', 'hessian')):
        """
        Initialization method.
        """
        self._unique_kbody_terms = None
        self._kbody_terms = None

        super(MeamNN, self).__init__(
            elements=elements,
            custom_potentials=custom_potentials,
            hidden_sizes=hidden_sizes,
            activation=activation,
            minimize_properties=minimize_properties,
            export_properties=export_properties)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        return super(MeamNN, self).as_dict()

    def _get_model_outputs(self,
                           features: AttributeDict,
                           descriptors: AttributeDict,
                           mode: tf_estimator.ModeKeys,
                           verbose=False):
        """
        Return raw NN-EAM model outputs.

        Parameters
        ----------
        features : AttributeDict
            A dict of tensors, includeing raw properties and the descriptors:
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        descriptors : AttributeDict
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
        with tf.variable_scope("MEAM"):

            partitions, max_occurs = self._dynamic_partition(
                descriptors=descriptors,
                mode=mode,
                merge_symmetric=False)

            rho, _ = self._build_rho_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            embed = self._build_embed_nn(
                rho=rho,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            partitions, max_occurs = self._dynamic_partition(
                descriptors=descriptors,
                mode=mode,
                merge_symmetric=True)

            phi, _ = self._build_phi_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            y = tf.add(phi, embed, name='atomic')

            return y
