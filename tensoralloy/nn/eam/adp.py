#!coding=utf-8
"""
This module defines ADP potential.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

import tensorflow as tf
import numpy as np

from tensorflow_estimator import estimator as tf_estimator
from collections import Counter
from functools import partial

from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.utils import AttributeDict, get_elements_from_kbody_term
from tensoralloy.utils import get_kbody_terms, Defaults, safe_select


class AdpNN(EamAlloyNN):
    """
    The ADP model is just an extension of the EAM/Alloy model. Dipole and
    quadrupole interactions are introduced to the ADP model.
    """

    @property
    def tag(self):
        """ Return the tag. """
        return "alloy"

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
        kbody_terms = get_kbody_terms(self._elements, k_max=2)[1]
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
            results[element] = {'rho': Defaults.hidden_sizes,
                                'embed': Defaults.hidden_sizes}
            _safe_update(element)

        for kbody_term in unique_kbody_terms:
            results[kbody_term] = {'phi': Defaults.hidden_sizes,
                                   'dipole': Defaults.hidden_sizes,
                                   'quadrupole': Defaults.hidden_sizes}
            _safe_update(kbody_term)

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

        if isinstance(custom_potentials, str):
            potentials = {el: {"rho": custom_potentials,
                               "embed": custom_potentials}
                          for el in self._elements}
            potentials.update({kbody_term: {"phi": custom_potentials,
                                            "dipole": custom_potentials,
                                            "quadrupole": custom_potentials}
                               for kbody_term in self._unique_kbody_terms})
            return potentials

        potentials = {el: {"rho": "nn", "embed": "nn"} for el in self._elements}
        potentials.update({kbody_term: {"phi": "nn",
                                        "dipole": "nn",
                                        "quadrupole": "nn"}
                           for kbody_term in self._unique_kbody_terms})

        custom_potentials = safe_select(custom_potentials, {})

        def _safe_update(section, key):
            if key in custom_potentials[section]:
                value = custom_potentials[section][key]
                assert _check_avail(value)
                potentials[section][key] = value

        for element in self._elements:
            if element in custom_potentials:
                _safe_update(element, 'rho')
                _safe_update(element, 'embed')

        for kbody_term in self._unique_kbody_terms:
            if kbody_term in custom_potentials:
                _safe_update(kbody_term, 'phi')
                _safe_update(kbody_term, 'dipole')
                _safe_update(kbody_term, 'quadrupole')

        return potentials

    def _get_dipole_fn(self, kbody_term: str, variable_scope='Dipole',
                       verbose=False):
        """
        Return the pairwise dipole function of `name` for the given k-body
        term.
        """
        name = self._potentials[kbody_term]['dipole']
        if name == 'nn':
            return self._get_nn_fn(
                section=kbody_term,
                key='dipole',
                variable_scope=f"{variable_scope}/{kbody_term}",
                verbose=verbose)
        else:
            return partial(self._empirical_functions[name].dipole,
                           kbody_term=kbody_term,
                           variable_scope=variable_scope,
                           verbose=verbose)

    def _get_quadrupole_fn(self, kbody_term: str, variable_scope='Quadrupole',
                           verbose=False):
        """
        Return the pairwise quadrupole function of `name` for the given k-body
        term.
        """
        name = self._potentials[kbody_term]['quadrupole']
        if name == 'nn':
            return self._get_nn_fn(
                section=kbody_term,
                key='quadrupole',
                variable_scope=f"{variable_scope}/{kbody_term}",
                verbose=verbose)
        else:
            return partial(self._empirical_functions[name].quadrupole,
                           kbody_term=kbody_term,
                           variable_scope=variable_scope,
                           verbose=verbose)

    def _build_rho_nn(self,
                      partitions: AttributeDict,
                      mode: tf_estimator.ModeKeys,
                      max_occurs: Counter,
                      verbose=False):
        """
        Return the outputs of the electron densities, `rho(r)`.

        Parameters
        ----------
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask.
            `mask` is a 4D tensor, `[batch_size, 1, max_n_element, nnl]`.
            `value` is a 5D tensor, `[4, batch_size, 1, max_n_element, nnl]`.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        atomic : tf.Tensor
            A 1D (PREDICT) or 2D (TRAIN or EVAL) tensor. The last axis has the
            size `max_n_atoms`.
        values : Dict[str, tf.Tensor]
            The corresponding value tensor of each `kbody_term` of
            `descriptors`. Each value tensor is a 5D tensor of shape
            `[batch_size, 1, max_n_element, nnl, 1]`. If `mode` is PREDICT,
            `batch_size` will be 1.

        """
        outputs = {}
        values = {}
        with tf.name_scope("Rho"):
            for kbody_term, (value, mask) in partitions.items():
                center, other = get_elements_from_kbody_term(kbody_term)
                with tf.name_scope(f"{kbody_term}"):
                    # In this class, `value[0]` corresponds to `value` in
                    # `EamAlloyNN._build_rho_nn`.
                    x = tf.expand_dims(value[0], axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    comput = self._get_rho_fn(other, verbose=verbose)
                    y = comput(x)
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')
                    values[kbody_term] = y
                    rho = tf.reduce_sum(
                        y, axis=(1, 3, 4), keepdims=False, name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[kbody_term] = rho
            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=False)
            if mode == tf_estimator.ModeKeys.PREDICT:
                atomic = tf.squeeze(atomic, axis=0)
            return atomic, values

    def _build_phi_nn(self, partitions: AttributeDict, max_occurs: Counter,
                      mode: tf_estimator.ModeKeys, verbose=False):
        """
        Return the outputs of the pairwise interactions, `Phi(r)`.

        Parameters
        ----------
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask.
            `mask` is a 4D tensor, `[batch_size, 1 + delta, max_n_el, nnl]`.
            `value` is a 5D tensor, `[4, batch_size, 1 + delta, max_n_el, nnl]`.
            `delta` will be zero if the corresponding kbody term has only one
            type of atom; otherwise `delta` will be one.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        atomic : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the energies
            of the real atoms.
        values : Dict[str, tf.Tensor]
            A dict. The corresponding value tensors before `reduce_sum` for
            `partitions`. Each value of `values` is a 5D tensor of shape
            `[batch_size, 1 + delta, max_n_element, nnl, 1]`. `delta` will be
            zero if the corresponding kbody term has only one type of atom;
            otherwise `delta` will be one.

        """
        outputs = {}
        values = {}
        with tf.name_scope("Phi"):
            for kbody_term, (value, mask) in partitions.items():
                with tf.name_scope(f"{kbody_term}"):
                    x = tf.expand_dims(value[0], axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    comput = self._get_phi_fn(kbody_term, verbose=verbose)
                    y = comput(x)
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')
                    values[kbody_term] = y
                    y = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                    y = tf.squeeze(y, axis=1)
                    half = tf.constant(0.5, dtype=x.dtype, name='half')
                    y = tf.multiply(y, half, name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs[kbody_term] = y
            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=True)
            if mode == tf_estimator.ModeKeys.PREDICT:
                atomic = tf.squeeze(atomic, axis=0, name='squeeze')
            return atomic, values

    def _build_dipole_nn(self, partitions: AttributeDict, max_occurs: Counter,
                         mode: tf_estimator.ModeKeys, verbose=False):
        """
        Return the outputs of the dipole interactions.

        Parameters
        ----------
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask.
            `mask` is a 4D tensor, `[batch_size, 1 + delta, max_n_el, nnl]`.
            `value` is a 5D tensor, `[4, batch_size, 1 + delta, max_n_el, nnl]`.
            `delta` will be zero if the corresponding kbody term has only one
            type of atom; otherwise `delta` will be one.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        atomic : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the energies
            of the real atoms.
        values : Dict[str, tf.Tensor]
            A dict. The corresponding value tensors before `reduce_sum` for
            `partitions`. Each value of `values` is a 5D tensor of shape
            `[batch_size, 1 + delta, max_n_element, nnl, 1]`. `delta` will be
            zero if the corresponding kbody term has only one type of atom;
            otherwise `delta` will be one.

        """
        outputs = {}
        values = {}
        with tf.name_scope("Dipole"):
            for kbody_term, (value, mask) in partitions.items():
                with tf.name_scope(f"{kbody_term}"):

                    rij = tf.expand_dims(value[0], axis=-1, name='input')
                    if verbose:
                        log_tensor(rij)

                    # Apply the dipole function on `rij`
                    comput = self._get_dipole_fn(kbody_term, verbose=verbose)
                    uij = comput(rij)

                    # Apply the value mask
                    uij = tf.multiply(uij, tf.expand_dims(mask, axis=-1),
                                      name='uij/masked')
                    values[kbody_term] = uij

                    # Compute the total dipole contribution
                    components = []
                    for ia, alpha in enumerate(('x', 'y', 'z')):
                        with tf.name_scope(f"{alpha}"):
                            dij = tf.identity(value[ia + 1], name=f'd{alpha}')
                            udaij = tf.multiply(uij, dij, name='udaij')
                            uda = tf.reduce_sum(udaij,
                                                axis=(3, 4),
                                                keep_dims=False,
                                                name='uda')
                            uda2 = tf.square(uda, name='uda2')
                            components.append(uda2)
                    y = tf.add_n(components)

                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    y = tf.squeeze(y, axis=1)
                    half = tf.constant(0.5, dtype=rij.dtype, name='half')
                    y = tf.multiply(y, half, name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs[kbody_term] = y
            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=True)
            if mode == tf_estimator.ModeKeys.PREDICT:
                atomic = tf.squeeze(atomic, axis=0, name='squeeze')
            return atomic, values

    def _build_quadrupole_nn(self,
                             partitions: AttributeDict,
                             max_occurs: Counter,
                             mode: tf_estimator.ModeKeys,
                             verbose=False):
        """
        Return the outputs of the quadrupole interactions.

        Parameters
        ----------
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask.
            `mask` is a 4D tensor, `[batch_size, 1 + delta, max_n_el, nnl]`.
            `value` is a 5D tensor, `[4, batch_size, 1 + delta, max_n_el, nnl]`.
            `delta` will be zero if the corresponding kbody term has only one
            type of atom; otherwise `delta` will be one.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        atomic : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the energies
            of the real atoms.
        values : Dict[str, tf.Tensor]
            A dict. The corresponding value tensors before `reduce_sum` for
            `partitions`. Each value of `values` is a 5D tensor of shape
            `[batch_size, 1 + delta, max_n_element, nnl, 1]`. `delta` will be
            zero if the corresponding kbody term has only one type of atom;
            otherwise `delta` will be one.

        """
        outputs = {}
        values = {}
        with tf.name_scope("Dipole"):
            for kbody_term, (value, mask) in partitions.items():
                with tf.name_scope(f"{kbody_term}"):

                    rij = tf.expand_dims(value[0], axis=-1, name='input')
                    if verbose:
                        log_tensor(rij)

                    # Apply the dipole function on `rij`
                    comput = self._get_quadrupole_fn(
                        kbody_term, verbose=verbose)
                    wij = comput(rij)

                    # Apply the value mask
                    wij = tf.multiply(wij, tf.expand_dims(mask, axis=-1),
                                      name='wij/masked')
                    values[kbody_term] = wij

                    # Compute the total dipole contribution
                    components = []
                    diag = []
                    for ia, alpha in enumerate(('x', 'y', 'z')):
                        for ib, beta in enumerate(('x', 'y', 'z')):
                            with tf.name_scope(f"{alpha}{beta}"):
                                dija = tf.identity(value[ia + 1], f'd{alpha}')
                                dijb = tf.identity(value[ib + 1], f'd{beta}')
                                wdabij = tf.multiply(wij, dija * dijb, 'udabij')
                                wdab = tf.reduce_sum(wdabij,
                                                     axis=(3, 4),
                                                     keep_dims=False,
                                                     name='udab')
                                wdab2 = tf.square(wdab, name='udab2')
                                components.append(wdab2)
                                if alpha == beta:
                                    diag.append(wdab2)
                    y_full = tf.add_n(components)
                    y_diag = tf.add_n(diag)

                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    y_full = tf.squeeze(y_full, axis=1)
                    y_diag = tf.squeeze(y_diag, axis=1)
                    half = tf.constant(0.5, dtype=rij.dtype, name='half')
                    three = tf.constant(3.0, dtype=rij.dtype, name='three')
                    y = tf.math.subtract(half * y_full,
                                         half * y_diag / three,
                                         name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs[kbody_term] = y
            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=True)
            if mode == tf_estimator.ModeKeys.PREDICT:
                atomic = tf.squeeze(atomic, axis=0, name='squeeze')
            return atomic, values

    def _dynamic_partition(self,
                           descriptors: AttributeDict,
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
        descriptors : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are elements and values are tuples of (value, mask)
            where where `value` represents the descriptors and `mask` is
            the value mask. `value` and `mask` have the same shape.
                * If `mode` is TRAIN or EVAL, both should be 4D tensors of shape
                  `[4, batch_size, max_n_terms, max_n_element, nnl]`.
                * If `mode` is PREDICT, both should be 3D tensors of shape
                  `[4, max_n_terms, max_n_element, nnl]`.
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
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, 1 + delta, max_n_element, nnl]`. `delta` will be zero
            if the corresponding kbody term has only one type of atom; otherwise
            `delta` will be one.
        max_occurs : Counter
            The maximum occurance of each type of element.

        """
        partitions = AttributeDict()
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
                        masks = tf.expand_dims(masks, axis=0)
                        max_occurs[element] = tf.shape(values)[2]
                        g_axis = 1
                    else:
                        assert values.shape.ndims == 5
                        max_occurs[element] = values.shape[3].value
                        g_axis = 2

                    num = len(kbody_terms)
                    glists = tf.split(
                        values, num_or_size_splits=num, axis=g_axis,
                        name='glist')
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
                                    (partitions[kbody_term][0], value), axis=2)
                                mask = tf.concat(
                                    (partitions[kbody_term][1], mask), axis=2)
                        partitions[kbody_term] = (value, mask)
            return partitions, Counter(max_occurs)

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
                * 'cells' of shape `[batch_size, 3, 3]`.
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
        with tf.variable_scope("nnEAM"):
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

            dipole, _ = self._build_dipole_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose
            )

            quadrupole, _ = self._build_quadrupole_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose,
            )

            y = tf.add_n([phi, embed, dipole, quadrupole], name='atomic')

            return y

    def export_to_setfl(self, setfl: str, nr: int, dr: float, nrho: int,
                        drho: float, checkpoint=None, lattice_constants=None,
                        lattice_types=None):
        """
        Export this ADP model to a setfl potential file for LAMMPS.

        Parameters
        ----------
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
        checkpoint : str or None
            The tensorflow checkpoint file to restore. If None, the default
            (or initital) parameters will be used.
        lattice_constants : Dict[str, float] or None
            The lattice constant for each type of element.
        lattice_types : Dict[str, str] or None
            The lattice type, e.g 'fcc', for each type of element.

        """
        # TODO: fix this function
        super(AdpNN, self).export_to_setfl(setfl, nr, dr, nrho, drho,
                                           checkpoint, lattice_constants,
                                           lattice_types)
