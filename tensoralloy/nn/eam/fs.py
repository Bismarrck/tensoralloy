# coding=utf-8
"""
This module defines the EAM/Finnis-Sinclair model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from collections import Counter
from os.path import join, dirname
from datetime import datetime
from ase.data import atomic_masses, atomic_numbers
from atsim.potentials import writeSetFLFinnisSinclair, Potential, EAMPotential
from typing import Dict, List

from tensoralloy.utils import get_kbody_terms, get_elements_from_kbody_term
from tensoralloy.utils import GraphKeys, Defaults, safe_select, ModeKeys
from tensoralloy.nn.utils import log_tensor
from tensoralloy.nn.eam.eam import EamNN, plot_potential
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EamFsNN(EamNN):
    """
    The tensorflow based implementation of the EAM/Finnis-Sinclair model.
    """

    default_collection = GraphKeys.EAM_FS_NN_VARIABLES

    @property
    def all_kbody_terms(self) -> List[str]:
        """
        Return a list of str as all the ordered k-body terms.
        """
        return self._all_kbody_terms

    @property
    def tag(self):
        """ Return the tag. """
        return "fs"

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
            get_kbody_terms(self._elements, angular=False)[:2]
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
        if isinstance(custom_potentials, str):
            potentials = {el: {"embed": custom_potentials}
                          for el in self._elements}
            potentials.update({kbody_term: {"phi": custom_potentials,
                                            "rho": custom_potentials}
                               for kbody_term in self._all_kbody_terms})

            return potentials

        potentials = {el: {"embed": "nn"} for el in self._elements}
        potentials.update({kbody_term: {"phi": "nn", "rho": "nn"}
                           for kbody_term in self._all_kbody_terms})

        custom_potentials = safe_select(custom_potentials, {})

        def _safe_update(section, key):
            if key in custom_potentials[section]:
                value = custom_potentials[section][key]
                assert self._check_fn_avail(value)
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
        with tf.variable_scope("Rho"):
            for kbody_term, (value, mask) in partitions.items():
                with tf.variable_scope(f"{kbody_term}"):
                    x = tf.squeeze(value[0], axis=1, name='input')
                    if verbose:
                        log_tensor(x)
                    # Apply the `rho` function on `x`
                    comput = self._get_rho_fn(kbody_term, verbose=verbose)
                    y = comput(x)
                    # Apply the mask to rho.
                    y = tf.multiply(y, tf.squeeze(mask, axis=1),
                                    name='masked')
                    values[kbody_term] = y
                    rho = tf.reduce_sum(
                        y, axis=(2, 3), keepdims=False, name='rho')
                    if verbose:
                        log_tensor(rho)
                    outputs[kbody_term] = rho
            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=False)
            if mode == ModeKeys.PREDICT or mode == ModeKeys.LAMMPS:
                atomic = tf.squeeze(atomic, axis=0)
            return atomic, values

    def export_to_setfl(self, setfl: str, nr: int, dr: float, nrho: int,
                        drho: float, r0=1.0, rt=None, rho0=0.0, rhot=None,
                        checkpoint=None, lattice_constants=None,
                        lattice_types=None, use_ema_variables=True):
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
        r0 : float
            The initial `r` for plotting density and pair potentials.
        rt : float
            The final `r` for plotting density and pair potentials.
        rho0 : float
            The initial `rho` for plotting embedding functions.
        rhot : float
            The final `rho` for plotting embedding functions.
        checkpoint : str
            The tensorflow checkpoint file to restore.
        lattice_constants : Dict[str, float] or None
            The lattice constant for each type of element.
        lattice_types : Dict[str, str] or None
            The lattice type, e.g 'fcc', for each type of element.
        use_ema_variables : bool
            If True, exponentially moving averaged variables will be used.

        """
        dtype = get_float_dtype().as_numpy_dtype
        outdir = dirname(setfl)
        elements = self._elements
        rho = np.tile(np.arange(0.0, nrho * drho, drho, dtype=dtype),
                      reps=len(elements))
        rho = np.atleast_2d(rho)
        r = np.arange(
            0.0, nr * dr, dr, dtype=dtype).reshape((1, 1, 1, 1, -1, 1))
        lattice_constants = safe_select(lattice_constants, {})
        lattice_types = safe_select(lattice_types, {})
        mode = ModeKeys.EVAL

        with tf.Graph().as_default():

            with tf.name_scope("Inputs"):
                rho = tf.convert_to_tensor(rho, name='rho')

                partitions = dict()
                symmetric_partitions = dict()
                for kbody_term in self._all_kbody_terms:
                    value = tf.convert_to_tensor(r, name=f'r{kbody_term}')
                    mask = tf.ones_like(value[0], name=f'm{kbody_term}')
                    partitions[kbody_term] = (value, mask)
                    a, b = get_elements_from_kbody_term(kbody_term)
                    if a == b:
                        symmetric_partitions[kbody_term] = (value, mask)
                    elif kbody_term in self._unique_kbody_terms:
                        rr = np.concatenate((r, r), axis=3)
                        value = tf.convert_to_tensor(rr, name=f'r{kbody_term}')
                        mask = tf.ones_like(value[0], name=f'm{kbody_term}')
                        symmetric_partitions[kbody_term] = (value, mask)

            with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
                embed_vals = self._build_embed_nn(
                    rho,
                    max_occurs=Counter({el: nrho for el in elements}),
                    mode=mode,
                    verbose=False)
                _, rho_vals = self._build_rho_nn(
                    partitions,
                    max_occurs=Counter({el: 1 for el in elements}),
                    mode=mode,
                    verbose=False)
                _, phi_vals = self._build_phi_nn(
                    symmetric_partitions,
                    max_occurs=Counter({el: 1 for el in elements}),
                    mode=mode,
                    verbose=False)

            sess = tf.Session()
            with sess:
                tf.global_variables_initializer().run()
                if checkpoint is not None:
                    if use_ema_variables:
                        # Restore the moving averaged variables
                        ema = tf.train.ExponentialMovingAverage(
                            Defaults.variable_moving_average_decay)
                        saver = tf.train.Saver(ema.variables_to_restore())
                    else:
                        saver = tf.train.Saver(tf.trainable_variables())
                    saver.restore(sess, checkpoint)

                results = sess.run(
                    {'embed': embed_vals, 'rho': rho_vals, 'phi': phi_vals})

                def make_density(_kbody_term):
                    """
                    Return the density function rho(r) for `element`.
                    """
                    def _func(_r):
                        """ Return `rho(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results["rho"][_kbody_term][0, 0, idx, 0]
                    return _func

                def make_embed(_element):
                    """
                    Return the embedding energy function F(rho) for `element`.
                    """
                    base = nr * self._elements.index(_element)

                    def _func(_rho):
                        """ Return `F(rho)` for the given `rho`. """
                        idx = int(round(_rho / drho, 6))
                        return results["embed"][0, base + idx]
                    return _func

                def make_pairwise(_kbody_term):
                    """
                    Return the embedding energy function phi(r) for `element`.
                    """
                    def _func(_r):
                        """ Return `phi(r)` for the given `r`. """
                        idx = int(round(_r / dr, 6))
                        return results["phi"][_kbody_term][0, 0, idx, 0]
                    return _func

            eam_potentials = []
            for element in self._elements:
                number = atomic_numbers[element]
                mass = atomic_masses[number]
                embed_fn = make_embed(element)
                density_potentials = {}
                for kbody_term in self._kbody_terms[element]:
                    specie = get_elements_from_kbody_term(kbody_term)[1]
                    density_fn = make_density(kbody_term)
                    density_potentials[specie] = density_fn

                    plot_potential(nr, dr, density_fn, x0=r0, xt=rt,
                                   filename=join(outdir,
                                                 f"{kbody_term}.rho.png"),
                                   xlabel=r"$r (\AA)$",
                                   ylabel=r"$\mathbf{\rho}(r)$ (eV)",
                                   title=r"$\mathbf{\rho}(r)$ of " + kbody_term)

                potential = EAMPotential(
                    element, number, mass, embed_fn,
                    density_potentials,
                    latticeConstant=lattice_constants.get(element, 0.0),
                    latticeType=lattice_types.get(element, 'fcc'),
                )
                eam_potentials.append(potential)

                plot_potential(nrho, drho, embed_fn, x0=rho0, xt=rhot,
                               filename=join(outdir, f"{element}.embed.png"),
                               xlabel=r"$\rho$",
                               ylabel=r"$\mathbf{F}(\rho)$ (eV)",
                               title=r"$\mathbf{F}(\rho)$ of " + element)

            pair_potentials = []
            for kbody_term in self._unique_kbody_terms:
                a, b = get_elements_from_kbody_term(kbody_term)
                pairwise_fn = make_pairwise(kbody_term)
                potential = Potential(a, b, pairwise_fn)
                pair_potentials.append(potential)

                plot_potential(nr, dr, pairwise_fn, x0=r0, xt=rt,
                               filename=join(outdir, f"{kbody_term}.phi.png"),
                               xlabel=r"$r (\AA)$",
                               ylabel=r"$\mathbf{\phi}(r)$ (eV)",
                               title=r"$\mathbf{\phi}(r)$ of " + kbody_term)

            comments = [
                "Date: {} Contributor: Xin Chen (Bismarrck@me.com)".format(
                    datetime.today()),
                "LAMMPS setfl format",
                "Conversion by TensorAlloy.nn.eam.fs.EamAlloyNN"
            ]

            with open(setfl, 'wb') as fp:
                writeSetFLFinnisSinclair(
                    nrho, drho, nr, dr, eam_potentials, pair_potentials, out=fp,
                    comments=comments)

            tf.logging.info(f"Model exported to {setfl}")
