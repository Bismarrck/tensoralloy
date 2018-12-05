# coding=utf-8
"""
This module defines the EAM/Finnis-Sinclair model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from collections import Counter

from tensoralloy.misc import safe_select, Defaults, AttributeDict
from tensoralloy.utils import get_kbody_terms, get_elements_from_kbody_term
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.eam.eam import EamNN
from tensoralloy.nn.eam.potentials import available_potentials

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EamFsNN(EamNN):
    """
    The tensorflow based implementation of the EAM/Finnis-Sinclair model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization method.
        """
        super(EamFsNN, self).__init__(*args, **kwargs)

    def _get_hidden_sizes(self, hidden_sizes):
        """
        Setup the hidden layer sizes.

        Parameters
        ----------
        hidden_sizes : int or List[int] or Dict[str, Dict[str, List[int]]]
            This can be an int, a list of int or a nested dict.

        Returns
        -------
        results : Dict[str, Dict[str, int]]
            A nested dict.

        """
        all_kbody_terms, kbody_terms = \
            get_kbody_terms(self._elements, k_max=2)[:2]
        hidden_sizes = safe_select(hidden_sizes, Defaults.hidden_sizes)

        unique_kbody_terms = []
        for element in self._elements:
            for kbody_term in kbody_terms[element]:
                a, b = get_elements_from_kbody_term(kbody_term)
                if a == b:
                    unique_kbody_terms.append(kbody_term)
                else:
                    ab = "".join(sorted([a, b]))
                    if ab not in unique_kbody_terms:
                        unique_kbody_terms.append(ab)

        self._unique_kbody_terms = unique_kbody_terms
        self._all_kbody_terms = all_kbody_terms
        self._kbody_terms = kbody_terms

        results = {}

        def _safe_update(section):
            if isinstance(hidden_sizes, dict):
                if section in hidden_sizes:
                    results[section].update(hidden_sizes[section])
            else:
                value = np.atleast_1d(hidden_sizes).tolist()
                for key in results[section]:
                    results[section][key] = value

        for element in self._elements:
            results[element] = {'embed': Defaults.hidden_sizes}
            _safe_update(element)

        for kbody_term in all_kbody_terms:
            results[kbody_term] = {'phi': Defaults.hidden_sizes,
                                   'rho': Defaults.hidden_sizes}
            _safe_update(kbody_term)

            if kbody_term not in unique_kbody_terms:
                del results[kbody_term]['phi']

        return results

    def _setup_potentials(self, custom_potentials=None):
        """
        Setup the layers for nn-EAM.
        """

        def _check_avail(name: str):
            name = name.lower()
            if name == "nn" or name in available_potentials:
                return True
            else:
                return False

        potentials = {el: {"rho": "nn", "embed": "nn"} for el in self._elements}
        potentials.update({kbody_term: {"phi": "nn"}
                           for kbody_term in self._unique_kbody_terms})

        custom_potentials = safe_select(custom_potentials, {})

        def _safe_update(section, key):
            if key in custom_potentials[section]:
                value = custom_potentials[section][key]
                assert _check_avail(value)
                potentials[section][key] = value

        for element in self._elements:
            if element in custom_potentials:
                _safe_update(element, 'embed')

        for kbody_term in self._all_kbody_terms:
            if kbody_term in custom_potentials:
                _safe_update(kbody_term, 'phi')
                _safe_update(kbody_term, 'rho')

        return potentials

    def _build_rho_nn(self, partitions: AttributeDict, verbose=False):
        """
        Return the outputs of the electron densities, `rho(r)`.

        Parameters
        ----------
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, 1, max_n_element, nnl]`.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        rho : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]`.

        """
        outputs = {}
        with tf.name_scope("Rho"):
            for kbody_term, (value, mask) in partitions.items():
                with tf.variable_scope(kbody_term):
                    x = tf.expand_dims(value, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    # Apply the `rho` function on `x`
                    comput = self._get_rho_fn(kbody_term, verbose=verbose)
                    y = comput(x)
                    # Apply the mask to rho.
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')

                    y = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                    rho = tf.squeeze(y, axis=1, name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[kbody_term] = rho
            return self._dynamic_stitch(outputs, symmetric=False)

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Return the EAM/Alloy model.

        Parameters
        ----------
        features : AttributeDict
            A dict of tensors:
                * 'descriptors', a dict of (element, (value, mask)) where
                  `element` represents the symbol of an element, `value` is the
                  descriptors of `element` and `mask` is the mask of `value`.
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.

        """
        with tf.name_scope("nnEAM"):
            partitions, max_occurs = self._dynamic_partition(
                features, merge_symmetric=False)
            rho = self._build_rho_nn(partitions, verbose=verbose)
            embed = self._build_embed_nn(rho, max_occurs, verbose=verbose)

            partitions, max_occurs = self._dynamic_partition(
                features, merge_symmetric=True)
            phi = self._build_phi_nn(partitions, verbose=verbose)
            y = tf.add(phi, embed, name='atomic')
            return y

    def export(self, checkpoint_path: str, setfl: str, nr: int, dr: float,
               nrho: int, drho: float):
        """
        Export this EAM/Alloy model to a setfl potential file for LAMMPS.

        Parameters
        ----------
        checkpoint_path : str
            The tensorflow checkpoint file to restore.
        setfl : str
            The setfl file to write.
        nr : int
            The number of `r` used to describe density and pair potentials.
        dr : float
            The delta `r` used for tabulating density and pair potentials.
        nrho : int
            The number of `rho` used to describe embedding functions.
        drho : float
            The delta `rho` used for tabulating embedding functions.

        """
        rho = np.tile(np.arange(0.0, nrho * drho, drho, dtype=np.float64),
                      reps=len(self._elements))
        r = np.arange(0.0, nr * dr, nr).reshape((1, 1, 1, -1))

        max_occurs = Counter({el: nrho for el in self._elements})
