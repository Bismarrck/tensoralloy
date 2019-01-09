# coding=utf-8
"""
This module defines the EAM/Alloy model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from datetime import datetime
from os.path import dirname, join
from ase import data
from collections import Counter
from typing import List, Dict, Tuple
from atsim.potentials import Potential, EAMPotential, writeSetFL

from tensoralloy.misc import Defaults, safe_select, AttributeDict
from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms
from tensoralloy.nn.utils import log_tensor, GraphKeys
from tensoralloy.nn.eam.eam import EamNN, plot_potential
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.dtypes import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EamAlloyNN(EamNN):
    """
    The tensorflow based implementation of the EAM/Alloy model.
    """

    default_collection = GraphKeys.EAM_ALLOY_NN_VARIABLES

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

    def _build_rho_nn(self, descriptors: AttributeDict, max_occurs: Counter,
                      mode: tf.estimator.ModeKeys, verbose=False):
        """
        Return the outputs of the electron densities, `rho(r)`.

        Parameters
        ----------
        descriptors : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are elements and values are tuples of (value, mask)
            where where `value` represents the interatomic distances and `mask`
            is the value mask. `value` and `mask` have the same shape.
                * If `mode` is TRAIN or EVAL, both should be 4D tensors of shape
                  `[batch_size, max_n_terms, max_n_element, nnl]`.
                * If `mode` is PREDICT, both should be 3D tensors of shape
                  `[max_n_terms, max_n_element, nnl]`.
        max_occurs : Counter
            The maximum occurance of each type of element. This arg is not
            needed here.
        mode : tf.estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        atomic : tf.Tensor
            A 1D (PREDICT) or 2D (TRAIN or EVAL) tensor. The last axis has the
            size `n_atoms_max`.
        values : Dict[str, tf.Tensor]
            The corresponding value tensor of each element of `descriptors`.
            Each value tensor is a 4D or 5D tensor.

        """
        outputs = {}
        values = {}

        if mode == tf.estimator.ModeKeys.PREDICT:
            axes = (0, 2, 3)
            concat_axis = 0
        else:
            axes = (1, 3, 4)
            concat_axis = 1

        with tf.name_scope("Rho"):
            for element, (value, mask) in descriptors.items():
                with tf.variable_scope(f'{element}/Rho'):
                    x = tf.expand_dims(value, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    # Apply the `rho` function on `x`
                    comput = self._get_rho_fn(element, verbose=verbose)
                    y = comput(x)
                    # Apply the mask to rho.
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')
                    values[element] = y
                    rho = tf.reduce_sum(y, axis=axes, keepdims=False,
                                        name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[element] = rho
            rho = tf.concat([outputs[el] for el in self._elements],
                            axis=concat_axis, name='atomic')
            return rho, values

    def _build_nn(self, features: AttributeDict, mode: tf.estimator.ModeKeys,
                  verbose=False):
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
        mode : tf.estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.

        Returns
        -------
        y : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.

        """
        with tf.name_scope("nnEAM"):

            partitions, max_occurs = self._dynamic_partition(
                descriptors=features.descriptors,
                mode=mode,
                merge_symmetric=True)

            rho, _ = self._build_rho_nn(
                descriptors=features.descriptors,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            embed = self._build_embed_nn(
                rho=rho,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            phi, _ = self._build_phi_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            y = tf.add(phi, embed, name='atomic')

            return y

    def export_to_setfl(self, setfl: str, nr: int, dr: float, nrho: int,
                        drho: float, checkpoint=None, lattice_constants=None,
                        lattice_types=None):
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
        checkpoint : str or None
            The tensorflow checkpoint file to restore. If None, the default
            (or initital) parameters will be used.
        lattice_constants : Dict[str, float] or None
            The lattice constant for each type of element.
        lattice_types : Dict[str, str] or None
            The lattice type, e.g 'fcc', for each type of element.

        """
        outdir = dirname(setfl)
        rho = np.tile(np.arange(0.0, nrho * drho, drho,
                                dtype=get_float_dtype().as_numpy_dtype),
                      reps=len(self._elements))
        rho = np.atleast_2d(rho)
        r = np.arange(0.0, nr * dr, dr).reshape((1, 1, 1, -1))
        elements = self._elements
        lattice_constants = safe_select(lattice_constants, {})
        lattice_types = safe_select(lattice_types, {})

        with tf.Graph().as_default():

            with tf.name_scope("Inputs"):
                rho = tf.convert_to_tensor(rho, name='rho')

                descriptors = AttributeDict()
                for element in self._elements:
                    value = tf.convert_to_tensor(r, name=f'r{element}')
                    mask = tf.ones_like(value, name=f'm{element}')
                    descriptors[element] = (value, mask)

                partitions = AttributeDict()
                for kbody_term in self._unique_kbody_terms:
                    a, b = get_elements_from_kbody_term(kbody_term)
                    if a == b:
                        value = tf.convert_to_tensor(r, name=f'r{kbody_term}')
                        mask = tf.ones_like(value, name=f'm{kbody_term}')
                    else:
                        rr = np.concatenate((r, r), axis=2)
                        value = tf.convert_to_tensor(rr, name=f'r{kbody_term}')
                        mask = tf.ones_like(value, name=f'm{kbody_term}')
                    partitions[kbody_term] = (value, mask)

            with tf.name_scope("Model"):
                embed = self._build_embed_nn(
                    rho,
                    max_occurs=Counter({el: nrho for el in elements}),
                    mode=tf.estimator.ModeKeys.EVAL)
                _, rho = self._build_rho_nn(
                    descriptors,
                    max_occurs=Counter({el: 1 for el in elements}),
                    mode=tf.estimator.ModeKeys.EVAL)
                _, phi = self._build_phi_nn(
                    partitions,
                    max_occurs=Counter({el: 1 for el in elements}),
                    mode=tf.estimator.ModeKeys.EVAL)

            sess = tf.Session()
            with sess:
                if checkpoint is not None:
                    # Restore the moving averaged variables
                    ema = tf.train.ExponentialMovingAverage(
                        Defaults.variable_moving_average_decay)
                    saver = tf.train.Saver(ema.variables_to_restore())
                    saver.restore(sess, checkpoint)
                else:
                    tf.global_variables_initializer().run()

                results = AttributeDict(sess.run(
                    {'embed': embed, 'rho': rho, 'phi': phi}))

                def make_density(_element):
                    """
                    Return the density function for `element`.
                    """
                    def _func(_r):
                        """ Return `rho(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results.rho[_element][0, 0, 0, idx, 0]
                    return _func

                def make_embed(_element):
                    """
                    Return the embedding energy function for `element`.
                    """
                    base = nr * self._elements.index(_element)

                    def _func(_rho):
                        """ Return `F(rho)` for the given `rho`. """
                        idx = int(round(_rho / drho, 6))
                        return results.embed[0, base + idx]
                    return _func

                def make_pairwise(_kbody_term):
                    """
                    Return the embedding energy function for `element`.
                    """
                    def _func(_r):
                        """ Return `phi(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results.phi[_kbody_term][0, 0, 0, idx, 0]
                    return _func

            eam_potentials = []
            for element in self._elements:
                number = data.atomic_numbers[element]
                mass = data.atomic_masses[number]
                embed_fn = make_embed(element)
                density_fn = make_density(element)
                potential = EAMPotential(
                    element, number, mass, embed_fn, density_fn,
                    latticeConstant=lattice_constants.get(element, 0.0),
                    latticeType=lattice_types.get(element, 'fcc'))
                eam_potentials.append(potential)

                plot_potential(nrho, drho, embed_fn,
                               filename=join(outdir, f"{element}.embed.png"),
                               xlabel=r"$\rho$",
                               ylabel=r"$\mathbf{F}(\rho)$ (eV)",
                               title=r"$\mathbf{F}(\rho)$ of " + element)
                plot_potential(nr, dr, density_fn,
                               filename=join(outdir, f"{element}.rho.png"),
                               xlabel=r"$r (\AA)$",
                               ylabel=r"$\mathbf{\rho}(r)$ (eV)",
                               title=r"$\mathbf{\rho}(r)$ of " + element)

            pair_potentials = []
            for kbody_term in self._unique_kbody_terms:
                a, b = get_elements_from_kbody_term(kbody_term)
                pairwise_fn = make_pairwise(kbody_term)
                potential = Potential(a, b, pairwise_fn)
                pair_potentials.append(potential)

                plot_potential(nr, dr, pairwise_fn,
                               filename=join(outdir, f"{kbody_term}.phi.png"),
                               xlabel=r"$r (\AA)$",
                               ylabel=r"$\mathbf{\phi}(r)$ (eV)",
                               title=r"$\mathbf{\phi}(r)$ of " + kbody_term)

            comments = [
                "Date: {} Contributor: Xin Chen (Bismarrck@me.com)".format(
                    datetime.today()),
                "LAMMPS setfl format",
                "Conversion by TensorAlloy.nn.eam.alloy.EamAlloyNN"
            ]

            with open(setfl, 'wb') as fp:
                writeSetFL(nrho, drho, nr, dr, eam_potentials, pair_potentials,
                           out=fp, comments=comments)

            tf.logging.info(f"Model exported to {setfl}")
