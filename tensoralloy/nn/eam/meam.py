#!coding=utf-8
"""
This module defines the MeamNN.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from typing import List, Dict
from collections import Counter
from functools import partial

from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.precision import get_float_dtype
from tensoralloy.utils import AttributeDict, get_elements_from_kbody_term
from tensoralloy.utils import safe_select, Defaults, get_kbody_terms
from tensoralloy.utils import ModeKeys

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
            minimize_properties=minimize_properties,
            export_properties=export_properties)

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        return super(MeamNN, self).as_dict()

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
        kbody_terms = get_kbody_terms(self._elements, angular=False)[1]
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
                                'embed': Defaults.hidden_sizes,
                                'fs': Defaults.hidden_sizes}
            _safe_update(element)

        for kbody_term in unique_kbody_terms:
            results[kbody_term] = {'phi': Defaults.hidden_sizes,
                                   'gs': Defaults.hidden_sizes}
            _safe_update(kbody_term)

        return results

    def _setup_potentials(self, custom_potentials=None):
        """
        Setup the layers for nn-EAM.
        """
        if isinstance(custom_potentials, str):
            potentials = {el: {"rho": custom_potentials,
                               "embed": custom_potentials,
                               "fs": custom_potentials,}
                          for el in self._elements}
            potentials.update({kbody_term: {"phi": custom_potentials,
                                            "gs": custom_potentials}
                               for kbody_term in self._unique_kbody_terms})
            return potentials

        potentials = {el: {"rho": "nn", "embed": "nn", "fs": "nn"}
                      for el in self._elements}
        potentials.update({kbody_term: {"phi": "nn", "gs": "nn"}
                           for kbody_term in self._unique_kbody_terms})

        custom_potentials = safe_select(custom_potentials, {})

        def _safe_update(section, key):
            if key in custom_potentials[section]:
                value = custom_potentials[section][key]
                assert self._check_fn_avail(value)
                potentials[section][key] = value

        for element in self._elements:
            if element in custom_potentials:
                _safe_update(element, 'rho')
                _safe_update(element, 'embed')
                _safe_update(element, 'fs')

        for kbody_term in self._unique_kbody_terms:
            if kbody_term in custom_potentials:
                _safe_update(kbody_term, 'phi')
                _safe_update(kbody_term, 'gs')

        return potentials

    def _get_gs_fn(self, kbody_term: str, variable_scope='Gs', verbose=False):
        """
        Return the Gs function.
        """
        name = self._may_insert_spline_fn(self._potentials[kbody_term]['gs'])
        if name == 'nn':
            return self._get_nn_fn(
                section=kbody_term,
                key='gs',
                variable_scope=f"{variable_scope}/{kbody_term}",
                verbose=verbose)
        else:
            return partial(getattr(self._empirical_functions[name], "gs"),
                           kbody_term=kbody_term,
                           variable_scope=variable_scope,
                           verbose=verbose)

    def _get_fs_fn(self, element: str, variable_scope='Fs', verbose=False):
        """
        Return the fs(rij) function.
        """
        name = self._may_insert_spline_fn(self._potentials[element]['fs'])
        if name == 'nn':
            return self._get_nn_fn(
                section=element,
                key='fs',
                variable_scope=f"{variable_scope}/{element}",
                verbose=verbose)
        else:
            return partial(getattr(self._empirical_functions[name], "fs"),
                           element=element,
                           variable_scope=variable_scope,
                           verbose=verbose)

    def _build_rho_nn(self,
                      partitions: dict,
                      mode: ModeKeys,
                      max_occurs: Counter,
                      verbose=False):
        """
        Return the outputs of the electron densities, `rho(r)`.

        Parameters
        ----------
        partitions : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, 1, max_n_element, nnl]`.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : ModeKeys
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
            for kbody_term, (dists, masks) in partitions.items():

                # Split `kbody_term` to two elements: center, other
                center, other = get_elements_from_kbody_term(kbody_term)

                with tf.name_scope(f"{kbody_term}"):
                    x = tf.identity(dists[0], name='x')
                    if verbose:
                        log_tensor(x)

                    # Get the rho function.
                    comput = self._get_rho_fn(other, verbose=verbose)

                    # Apply the `rho` function of element `other` on `x`
                    y = comput(x)

                    # Apply the `mask` to `rho`.
                    y = tf.multiply(y, masks, name='masked')
                    values[kbody_term] = y

                    # Compute the sum of electron densities from different `r`.
                    rho = tf.reduce_sum(
                        y, axis=(1, 3, 4), keepdims=False, name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[kbody_term] = rho

            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=False)
            if mode == ModeKeys.PREDICT:
                atomic = tf.squeeze(atomic, axis=0)
            return atomic, values

    def _build_angular_rho_nn(self,
                              partitions: dict,
                              mode: ModeKeys,
                              max_occurs: Counter,
                              verbose=False):
        rho_dict: Dict = {}

        with tf.name_scope("Rho/Angular"):
            dtype = get_float_dtype()
            with tf.name_scope("Constants"):
                two = tf.constant(2.0, dtype=dtype, name='two')
                half = tf.constant(0.5, dtype=dtype, name='half')

            for kbody_term, (dists, masks) in partitions.items():
                symboli, symbolj, symbolk = \
                    get_elements_from_kbody_term(kbody_term)
                with tf.variable_scope(f"{kbody_term}"):
                    masks = tf.squeeze(masks, axis=1, name='masks')
                    rij = tf.squeeze(dists[0], axis=1, name='rij')
                    rik = tf.squeeze(dists[4], axis=1, name='rik')
                    rjk = tf.squeeze(dists[8], axis=1, name='rjk')
                    rij2 = tf.square(rij, name='rij2')
                    rik2 = tf.square(rik, name='rik2')
                    rjk2 = tf.square(rjk, name='rjk2')
                    upper = tf.math.subtract(
                        rij2 + rik2, rjk2, name='upper')
                    lower = tf.math.multiply(
                        tf.math.multiply(two, rij), rik, name='lower')
                    costheta = tf.math.divide_no_nan(
                        upper, lower, name='costheta')

                    with tf.name_scope("Gs"):
                        comput = self._get_gs_fn(
                            f"{symbolj}{symbolk}", verbose=verbose)
                        gs = comput(costheta)
                        gs = tf.math.multiply(gs, masks, name='gs/masked')

                    with tf.name_scope("Fij"):
                        fij_fn = self._get_fs_fn(symbolj, verbose=verbose)
                        fij = fij_fn(rij)
                        fij = tf.math.multiply(fij, masks, name='fij/masked')

                    with tf.name_scope("Fik"):
                        fik_fn = self._get_fs_fn(symbolk, verbose=verbose)
                        fik = fik_fn(rik)
                        fik = tf.math.multiply(fik, masks, name='fik/masked')

                    rho = tf.math.multiply(fij * fik, gs, name='rho/ijk')
                    rho = tf.reduce_sum(
                        rho, axis=-1, keepdims=True, name='rho/sum')
                    rho = tf.math.multiply(rho, half, name='rho/half')
                    # rho_dict[symboli] = rho_dict.get(symboli, []) + [rho]
                    return tf.reduce_sum(rho, axis=[-1, -2], keepdims=False, name='rho')

    def _build_phi_nn(self, partitions: dict, max_occurs: Counter,
                      mode: ModeKeys, verbose=False):
        """
        Return the outputs of the pairwise interactions, `Phi(r)`.

        Parameters
        ----------
        partitions : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, 1 + delta, max_n_element, nnl]`. `delta` will be zero
            if the corresponding kbody term has only one type of atom; otherwise
            `delta` will be one.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : ModeKeys
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
            for kbody_term, (dists, masks) in partitions.items():
                with tf.name_scope(f"{kbody_term}"):
                    # name_scope `x` to a 5D tensor.
                    x = tf.identity(dists[0], name='x')
                    if verbose:
                        log_tensor(x)
                    # Apply the `phi` function on `x`
                    comput = self._get_phi_fn(kbody_term, verbose=verbose)
                    y = comput(x)
                    # Apply the value mask
                    y = tf.multiply(y, masks, name='masked')

                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    values[kbody_term] = y
                    y = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                    y = tf.squeeze(y, axis=1)
                    half = tf.constant(0.5, dtype=x.dtype, name='half')
                    y = tf.multiply(y, half, name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs[kbody_term] = y
            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=True)
            return atomic, values

    def _build_embed_nn(self, rho: tf.Tensor, max_occurs: Counter,
                        mode: ModeKeys, verbose=True):
        """
        Return the embedding energy, `F(rho)`.

        Parameters
        ----------
        rho : tf.Tensor
            A 1D or 2D tensor as the electron densities of the atoms. The last
            axis has the size `n_atoms_max`.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        embed : tf.Tensor
            The embedding energy. Has the same shape with `rho`.

        """
        split_sizes = [max_occurs[el] for el in self._elements]
        split_axis = 1
        squeeze_axis = 2

        with tf.name_scope("Embed"):
            splits = tf.split(
                rho, num_or_size_splits=split_sizes, axis=split_axis)
            values = []
            zero = tf.constant(0.0, dtype=rho.dtype, name='zero')
            for i, element in enumerate(self._elements):
                with tf.name_scope(f"{element}"):
                    x = tf.expand_dims(splits[i], axis=-1, name=element)
                    if verbose:
                        log_tensor(x)
                    # Apply the embedding function on `x`
                    comput = self._get_embed_fn(element, verbose=verbose)
                    y = comput(x)
                    with tf.name_scope("Baseline"):
                        base = comput(zero)
                    y = tf.math.subtract(y, base, name='relative')
                    embed = tf.squeeze(y, axis=squeeze_axis, name='atomic')
                    if verbose:
                        log_tensor(embed)
                    values.append(embed)
            return tf.concat(values, axis=split_axis)

    def _get_model_outputs(self,
                           features: AttributeDict,
                           descriptors: AttributeDict,
                           mode: ModeKeys,
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
        mode : ModeKeys
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
            clf = self._transformer
            assert isinstance(clf, UniversalTransformer)

            with tf.name_scope("Radial"):
                radial, max_occurs = dynamic_partition(
                    dists_and_masks=descriptors["radial"],
                    elements=self._elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=False,
                    merge_symmetric=False)
                symm = dynamic_partition(
                    dists_and_masks=descriptors["radial"],
                    elements=self._elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=False,
                    merge_symmetric=True)[0]

            with tf.name_scope("Angular"):
                angular, angular_max_occurs = dynamic_partition(
                    dists_and_masks=descriptors["angular"],
                    elements=self._elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=True,
                    merge_symmetric=False)

            rrho, _ = self._build_rho_nn(
                partitions=radial,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            arho = self._build_angular_rho_nn(
                partitions=angular,
                mode=mode,
                max_occurs=angular_max_occurs,
                verbose=verbose)

            rho = tf.math.add(arho, rrho, name='meam/spline/rho')
            if verbose:
                log_tensor(rho)

            embed = self._build_embed_nn(
                rho=rho,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            phi, _ = self._build_phi_nn(
                partitions=symm,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            y = tf.add(phi, embed, name='atomic')
            if mode == ModeKeys.PREDICT:
                y = tf.squeeze(y, axis=0, name='atomic/squeeze')

            return y
