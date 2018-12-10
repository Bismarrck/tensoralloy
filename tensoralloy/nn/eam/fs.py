# coding=utf-8
"""
This module defines the EAM/Finnis-Sinclair model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from collections import Counter
from datetime import datetime
from ase.data import atomic_masses, atomic_numbers
from atsim.potentials import writeSetFLFinnisSinclair, Potential, EAMPotential
from typing import Dict, List

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

    @property
    def all_kbody_terms(self) -> List[str]:
        """
        Return a list of str as all the ordered k-body terms.
        """
        return self._all_kbody_terms

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

        potentials = {el: {"embed": "nn"} for el in self._elements}
        potentials.update({kbody_term: {"phi": "nn", "rho": "nn"}
                           for kbody_term in self._all_kbody_terms})

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
                _safe_update(kbody_term, 'rho')
            if kbody_term not in self._unique_kbody_terms:
                del potentials[kbody_term]['phi']

        for kbody_term in self._unique_kbody_terms:
            if kbody_term in custom_potentials:
                _safe_update(kbody_term, 'phi')

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
        atomic : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]`.
        values : Dict[str, tf.Tensor]
            The corresponding value tensor of each `kbody_term` of
            `descriptors`. Each value tensor is a 5D tensor of shape
            `[batch_size, 1, max_n_element, nnl, 1]`.

        """
        outputs = {}
        values = {}
        with tf.name_scope("Rho"):
            for kbody_term, (value, mask) in partitions.items():
                with tf.variable_scope(f"{kbody_term}/Rho"):
                    x = tf.expand_dims(value, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    # Apply the `rho` function on `x`
                    comput = self._get_rho_fn(kbody_term, verbose=verbose)
                    y = comput(x)
                    # Apply the mask to rho.
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')
                    values[kbody_term] = y
                    rho = tf.reduce_sum(
                        y, axis=(1, 3, 4), keepdims=False, name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[kbody_term] = rho
            return self._dynamic_stitch(outputs, symmetric=False), values

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
            rho, _ = self._build_rho_nn(partitions, verbose=verbose)
            embed = self._build_embed_nn(rho, max_occurs, verbose=verbose)

            partitions, max_occurs = self._dynamic_partition(
                features, merge_symmetric=True)
            phi, _ = self._build_phi_nn(partitions, verbose=verbose)
            y = tf.add(phi, embed, name='atomic')
            return y

    def export(self, setfl: str, nr: int, dr: float, nrho: int, drho: float,
               checkpoint=None, lattice_constants=None, lattice_types=None):
        """
        Export this EAM/Alloy model to a setfl potential file for LAMMPS.

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
        checkpoint : str
            The tensorflow checkpoint file to restore.
        lattice_constants : Dict[str, float] or None
            The lattice constant for each type of element.
        lattice_types : Dict[str, str] or None
            The lattice type, e.g 'fcc', for each type of element.

        """
        rho = np.tile(np.arange(0.0, nrho * drho, drho, dtype=np.float64),
                      reps=len(self._elements))
        rho = np.atleast_2d(rho)
        r = np.arange(0.0, nr * dr, dr).reshape((1, 1, 1, -1))
        max_occurs = Counter({el: nrho for el in self._elements})
        lattice_constants = safe_select(lattice_constants, {})
        lattice_types = safe_select(lattice_types, {})

        with tf.Graph().as_default():

            with tf.name_scope("Inputs"):
                rho = tf.convert_to_tensor(rho, name='rho')

                partitions = AttributeDict()
                symmetric_partitions = AttributeDict()
                for kbody_term in self._all_kbody_terms:
                    value = tf.convert_to_tensor(r, name=f'r{kbody_term}')
                    mask = tf.ones_like(value, name=f'm{kbody_term}')
                    partitions[kbody_term] = (value, mask)
                    a, b = get_elements_from_kbody_term(kbody_term)
                    if a == b:
                        symmetric_partitions[kbody_term] = (value, mask)
                    elif kbody_term in self._unique_kbody_terms:
                        rr = np.concatenate((r, r), axis=2)
                        value = tf.convert_to_tensor(rr, name=f'r{kbody_term}')
                        mask = tf.ones_like(value, name=f'm{kbody_term}')
                        symmetric_partitions[kbody_term] = (value, mask)

            with tf.name_scope("Model"):
                embed = self._build_embed_nn(
                    rho, max_occurs=max_occurs, verbose=False)
                _, rho_vals = self._build_rho_nn(
                    partitions, verbose=False)
                _, phi_vals = self._build_phi_nn(
                    symmetric_partitions, verbose=False)

            sess = tf.Session()
            with sess:
                if checkpoint is not None:
                    saver = tf.train.Saver()
                    saver.restore(sess, checkpoint)
                else:
                    tf.global_variables_initializer().run()

                results = AttributeDict(sess.run(
                    {'embed': embed, 'rho': rho_vals, 'phi': phi_vals}))

                def make_density(_kbody_term):
                    """
                    Return the density function rho(r) for `element`.
                    """
                    def _func(_r):
                        """ Return `rho(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results.rho[_kbody_term][0, 0, 0, idx, 0]
                    return _func

                def make_embed(_element):
                    """
                    Return the embedding energy function F(rho) for `element`.
                    """
                    base = nr * self._elements.index(_element)

                    def _func(_rho):
                        """ Return `F(rho)` for the given `rho`. """
                        idx = int(round(_rho / drho, 6))
                        return results.embed[0, base + idx]
                    return _func

                def make_pairwise(_kbody_term):
                    """
                    Return the embedding energy function phi(r) for `element`.
                    """
                    def _func(_r):
                        """ Return `phi(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results.phi[_kbody_term][0, 0, 0, idx, 0]
                    return _func

            eam_potentials = []
            for element in self._elements:
                number = atomic_numbers[element]
                mass = atomic_masses[number]
                density_potentials = {}
                for kbody_term in self._kbody_terms[element]:
                    specie = get_elements_from_kbody_term(kbody_term)[1]
                    density_potentials[specie] = make_density(kbody_term)

                potential = EAMPotential(
                    element, number, mass, make_embed(element),
                    density_potentials,
                    latticeConstant=lattice_constants.get(element, 0.0),
                    latticeType=lattice_types.get(element, 'fcc'),
                )
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
                "Conversion by TensorAlloy.nn.eam.alloy.EamAlloyNN"
            ]

            with open(setfl, 'wb') as fp:
                writeSetFLFinnisSinclair(
                    nrho, drho, nr, dr, eam_potentials, pair_potentials, out=fp,
                    comments=comments)
