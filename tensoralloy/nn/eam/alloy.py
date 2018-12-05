# coding=utf-8
"""
This module defines the EAM/Alloy model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from datetime import datetime
from ase import data
from collections import Counter
from typing import List, Dict, Tuple
from atsim.potentials import Potential, EAMPotential, writeSetFL

from tensoralloy.misc import Defaults, safe_select, AttributeDict
from tensoralloy.nn.utils import log_tensor
from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms
from tensoralloy.nn.eam.eam import EamNN
from tensoralloy.nn.eam.potentials import available_potentials

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EamAlloyNN(EamNN):
    """
    The tensorflow based implementation of the EAM/Alloy model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization method.
        """
        super(EamAlloyNN, self).__init__(*args, **kwargs)

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
            results[kbody_term] = {'phi': Defaults.hidden_sizes}
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
                _safe_update(element, 'rho')
                _safe_update(element, 'embed')

        for kbody_term in self._unique_kbody_terms:
            if kbody_term in custom_potentials:
                _safe_update(kbody_term, 'phi')

        return potentials

    def _build_rho_nn(self, descriptors: AttributeDict, verbose=False):
        """
        Return the outputs of the electron densities, `rho(r)`.

        Parameters
        ----------
        descriptors : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are elements and values are tuples of (value, mask)
            where where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, max_n_terms, 1, nnl]`.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        rho : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]`.

        """
        outputs = {}
        with tf.name_scope("Rho"):
            for element, (value, mask) in descriptors.items():
                with tf.variable_scope(element):
                    x = tf.expand_dims(value, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    # Apply the `rho` function on `x`
                    comput = self._get_rho_fn(element, verbose=verbose)
                    y = comput(x)
                    # Apply the mask to rho.
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')

                    y = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                    rho = tf.squeeze(y, axis=1, name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[element] = rho
            return self._dynamic_stitch(outputs)

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
                features, merge_symmetric=True)
            rho = self._build_rho_nn(features.descriptors, verbose=verbose)
            embed = self._build_embed_nn(rho, max_occurs, verbose=verbose)
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

        with tf.Graph().as_default():
            saver = tf.train.Saver()

            with tf.name_scope("Values"):
                rho = tf.convert_to_tensor(rho, name='rho')

                descriptors = AttributeDict()
                for element in self._elements:
                    value = tf.convert_to_tensor(r, name=f'r{element}')
                    mask = tf.ones_like(value, name=f'm{element}')
                    descriptors[element] = (value, mask)

                partitions = AttributeDict()
                for kbody_term in self._unique_kbody_terms:
                    value = tf.convert_to_tensor(r, name=f'r{kbody_term}')
                    mask = tf.ones_like(value, name=f'm{kbody_term}')
                    partitions[kbody_term] = (value, mask)

            sess = tf.Session()
            with sess:
                saver.restore(sess, checkpoint_path)

                embed = self._build_embed_nn(rho, max_occurs=max_occurs)
                rho = self._build_rho_nn(descriptors)
                phi = self._build_phi_nn(partitions)

                results = AttributeDict(sess.run(
                    {'embed': embed, 'rho': rho, 'phi': phi}))

                def make_density(_element):
                    """
                    Return the density function for `element`.
                    """
                    base = nr * self._elements.index(_element)

                    def _func(_r):
                        """ Return `rho(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results.rho[base + idx]
                    return _func

                def make_embed(_element):
                    """
                    Return the embedding energy function for `element`.
                    """
                    base = nr * self._elements.index(_element)

                    def _func(_rho):
                        """ Return `F(rho)` for the given `rho`. """
                        idx = int(round(_rho / drho, 6))
                        return results.embed[base + idx]
                    return _func

                def make_pairwise(_kbody_term):
                    """
                    Return the embedding energy function for `element`.
                    """
                    base = nr * self._unique_kbody_terms.index(_kbody_term)

                    def _func(_r):
                        """ Return `phi(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results.phi[base + idx]
                    return _func

            eam_potentials = []
            for element in self._elements:
                number = data.atomic_numbers[element]
                mass = data.atomic_masses[element]
                potential = EAMPotential(element, number, mass,
                                         make_embed(element),
                                         make_density(element))
                eam_potentials.append(potential)

            pair_potentials = []
            for kbody_term in self._unique_kbody_terms:
                a, b = get_elements_from_kbody_term(kbody_term)
                potential = Potential(a, b, make_pairwise(kbody_term))
                pair_potentials.append(potential)

            comments = [
                "Date: {} Contributor: Xin Chen (Bismarrck@me.com)".format(
                    datetime.today()),
                "LAMMPS setfl format",
                "Conversion by TensorAlloy"
            ]

            with open(setfl, 'wb') as fp:
                writeSetFL(nrho, drho, nr, dr, eam_potentials, pair_potentials,
                           out=fp, comments=comments)
