# coding=utf-8
"""
This module defines the basic EAM-NN model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import re

from matplotlib import pyplot as plt
from collections import Counter
from functools import partial
from typing import List, Dict, Callable
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.dataclasses import EnergyOps, EnergyOp
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.nn.eam.potentials import EamFSPotential, EamAlloyPotential
from tensoralloy.nn.eam.potentials.spline import CubicSplinePotential
from tensoralloy.nn.eam.potentials.spline import LinearlyExtendedSplinePotential

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


_spline_pot_patt = re.compile(r"^spline@(.*)")
_lext_spline_pot_patt = re.compile(r"^lspline@(.*)")


def plot_potential(nx: int, dx: float, func: Callable, filename: str,
                   x0=0.0, xt=None, xlabel=None, ylabel=None, title=None):
    """
    Plot an empirical or NN potential.

    Parameters
    ----------
    nx : int
        The number of points.
    dx : float
        The gap of two adjacent points.
    func : Callable
        The function to compute f(x).
    filename : str
        The name of the output image.
    x0 : float
        The initial `x`.
    xt : float
        The final `x`.
    xlabel : str
        The label of X axis.
    ylabel : str
        The label of Y axis.
    title : str
        The title of the figure.

    """
    fig = plt.figure(1, figsize=[6, 6])

    x0 = int(x0 / dx) * dx
    if xt is None:
        xt = nx * dx
    else:
        xt = min(nx * dx, xt)

    x = np.linspace(x0, xt, num=int((xt - x0) / dx), endpoint=False)
    y = [func(xi) for xi in x]

    plt.plot(x, y, 'r-', linewidth=0.8)

    if title:
        plt.title(title, fontsize=15, fontname='arial')
    if xlabel:
        plt.xlabel(xlabel, fontsize=13, fontname='arial')
    if ylabel:
        plt.ylabel(ylabel, fontsize=13, fontname='arial')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


class EamNN(BasicNN):
    """
    The tensorflow/CNN based implementation of the Embedded-Atom Method.
    """

    scope = "EAM"

    def __init__(self,
                 elements: List[str],
                 custom_potentials=None,
                 hidden_sizes=None,
                 fixed_functions=None,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces', 'stress')):
        """
        Initialization method.
        """
        self._fixed_functions = fixed_functions or []
        self._unique_kbody_terms = None
        self._kbody_terms = None

        super(EamNN, self).__init__(
            elements=elements,
            hidden_sizes=hidden_sizes,
            minimize_properties=minimize_properties,
            export_properties=export_properties)

        # Setup the potentials
        self._potentials = self._setup_potentials(custom_potentials)

        # Initialize these empirical functions.
        self._empirical_functions = {
            key: cls() for key, cls in available_potentials.items()}

        # Asserts
        assert self._kbody_terms and self._unique_kbody_terms

    @property
    def unique_kbody_terms(self):
        """
        Return a list of str as all the unique k-body terms for this model.
        """
        return self._unique_kbody_terms

    def _get_hidden_sizes(self, hidden_sizes):
        raise NotImplementedError(
            "This method must be overridden by its subclass")

    @property
    def tag(self) -> str:
        """
        Return a str ('alloy' or 'fs') as the tag of this class.
        """
        raise NotImplementedError(
            "This property must be overridden by its subclass")

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        return {"class": self.__class__.__name__,
                "elements": self._elements,
                "custom_potentials": self._potentials,
                "hidden_sizes": self._hidden_sizes,
                "fixed_functions": self._fixed_functions,
                "minimize_properties": self._minimize_properties,
                "export_properties": self._export_properties}

    @property
    def potentials(self):
        """
        Return the layers.
        """
        return self._potentials

    def _setup_potentials(self, custom_potentials=None):
        """
        Setup the layers for nn-EAM.
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass")

    @staticmethod
    def _check_fn_avail(name: str):
        """
        Check the availability of the potential.
        """
        name = name.lower()
        if name == "nn" or \
                name in available_potentials or \
                name.startswith("spline@") or \
                name.startswith("lspline@"):
            return True
        else:
            return False

    def _may_insert_spline_fn(self, name: str):
        """
        Insert the spline functions read from the given csv file.
        """

        for (patt, cls) in (
                (_spline_pot_patt, CubicSplinePotential),
                (_lext_spline_pot_patt, LinearlyExtendedSplinePotential)):
            m = patt.search(name.strip())
            if m:
                json = m.group(1)
                if json not in self._empirical_functions:
                    spline = cls(json)
                    self._empirical_functions[json] = spline
                return json
        return name

    def _get_nn_fn(self, section, key, variable_scope, verbose=False):
        """
        Return a layer function of `f(x)` where `f` is a 1x1 CNN.
        """
        activation_fn = get_activation_fn(self._activation)
        hidden_sizes = self._hidden_sizes[section][key]
        if self.default_collection is None:
            collections = None
        else:
            collections = [self.default_collection]
        return partial(convolution1x1,
                       activation_fn=activation_fn,
                       hidden_sizes=hidden_sizes,
                       l2_weight=1.0,
                       variable_scope=variable_scope,
                       collections=collections,
                       verbose=verbose)

    def _get_embed_fn(self, element: str, variable_scope='Embed',
                      verbose=False):
        """
        Return the embedding function of `name` for `element`.
        """
        kwargs = dict(element=element,
                      variable_scope=variable_scope,
                      verbose=verbose)
        name = self._may_insert_spline_fn(self._potentials[element]['embed'])
        if name == 'nn':
            return self._get_nn_fn(
                section=element,
                key='embed',
                variable_scope=f"{variable_scope}/{element}",
                verbose=verbose)
        else:
            fixed = f"{element}.embed" in self._fixed_functions
            kwargs["fixed"] = fixed
            return partial(self._empirical_functions[name].embed,
                           **kwargs)

    def _get_rho_fn(self, element_or_kbody_term: str, variable_scope='Rho',
                    verbose=False):
        """
        Return the electron density function of `name` for the given k-body
        term.
        """
        name = self._may_insert_spline_fn(
            self._potentials[element_or_kbody_term]['rho'])
        if name == 'nn':
            return self._get_nn_fn(
                section=element_or_kbody_term,
                key='rho',
                variable_scope=f"{variable_scope}/{element_or_kbody_term}",
                verbose=verbose)
        else:
            fixed = f"{element_or_kbody_term}.rho" in self._fixed_functions
            pot = self._empirical_functions[name]
            if isinstance(pot, EamAlloyPotential):
                return partial(pot.rho,
                               element=element_or_kbody_term,
                               variable_scope=variable_scope,
                               fixed=fixed,
                               verbose=verbose)
            elif isinstance(pot, EamFSPotential):
                return partial(pot.rho,
                               kbody_term=element_or_kbody_term,
                               variable_scope=variable_scope,
                               fixed=fixed,
                               verbose=verbose)
            else:
                raise ValueError(
                    f"Unknown EAM potential: {pot.__class__.__name__}")

    def _get_phi_fn(self, kbody_term: str, variable_scope='Phi', verbose=False):
        """
        Return the pairwise potential function of `name` for the given k-body
        term.
        """
        name = self._may_insert_spline_fn(self._potentials[kbody_term]['phi'])
        if name == 'nn':
            return self._get_nn_fn(
                section=kbody_term,
                key='phi',
                variable_scope=f"{variable_scope}/{kbody_term}",
                verbose=verbose)
        else:
            fixed = f"{kbody_term}.phi" in self._fixed_functions
            return partial(self._empirical_functions[name].phi,
                           kbody_term=kbody_term,
                           variable_scope=variable_scope,
                           fixed=fixed,
                           verbose=verbose)

    def _get_energy_ops(self, outputs: tf.Tensor, features: dict, verbose=True):
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.
        features : Dict
            A dict of input features.
        name : str
            The name of the output tensor.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        ops : EnergyOps
            The energy tensors.

        """
        atomic_energy = tf.identity(outputs, name='atomic/raw')
        ndims = features["atom_masks"].shape.ndims
        axis = ndims - 1
        with tf.name_scope("Mask"):
            mask = tf.split(
                features["atom_masks"], [1, -1], axis=axis, name='split')[1]
        atomic_energy = tf.multiply(atomic_energy, mask, name='atomic')
        energy = tf.reduce_sum(
            atomic_energy, axis=axis, keepdims=False, name='energy')
        if verbose:
            log_tensor(energy)
        return EnergyOps(energy=EnergyOp(energy, atomic_energy))

    def _build_phi_nn(self, partitions: dict, max_occurs: Counter,
                      mode: tf_estimator.ModeKeys, verbose=False):
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
                    # name_scope `x` to a 5D tensor.
                    x = tf.squeeze(value[0], axis=1, name='rij')
                    if verbose:
                        log_tensor(x)
                    # Apply the `phi` function on `x`
                    comput = self._get_phi_fn(kbody_term, verbose=verbose)
                    y = comput(x)
                    # Apply the value mask
                    y = tf.multiply(y, tf.squeeze(mask, axis=1),
                                    name='masked')

                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    values[kbody_term] = y
                    y = tf.reduce_sum(y, axis=(2, 3), keepdims=False)
                    half = tf.constant(0.5, dtype=x.dtype, name='half')
                    y = tf.multiply(y, half, name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs[kbody_term] = y
            atomic = self._dynamic_stitch(outputs, max_occurs, symmetric=True)
            if mode == tf_estimator.ModeKeys.PREDICT:
                atomic = tf.squeeze(atomic, axis=0, name='squeeze')
            return atomic, values

    def _build_rho_nn(self,
                      partitions: dict,
                      mode: tf_estimator.ModeKeys,
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
        raise NotImplementedError(
            "This method must be overridden by its subclass")

    def _build_embed_nn(self, rho: tf.Tensor, max_occurs: Counter,
                        mode: tf_estimator.ModeKeys, verbose=True):
        """
        Return the embedding energy, `F(rho)`.

        Parameters
        ----------
        rho : tf.Tensor
            A 1D or 2D tensor as the electron densities of the atoms. The last
            axis has the size `n_atoms_max`.
        max_occurs : Counter
            The maximum occurance of each type of element.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        embed : tf.Tensor
            The embedding energy. Has the same shape with `rho`.

        """
        split_sizes = [max_occurs[el] for el in self._elements]

        if mode == tf_estimator.ModeKeys.PREDICT:
            split_axis = 0
            squeeze_axis = 1
        else:
            split_axis = 1
            squeeze_axis = 2

        with tf.name_scope("Embed"):
            splits = tf.split(
                rho, num_or_size_splits=split_sizes, axis=split_axis)
            values = []
            for i, element in enumerate(self._elements):
                with tf.name_scope(f"{element}"):
                    x = tf.expand_dims(splits[i], axis=-1, name=element)
                    if verbose:
                        log_tensor(x)
                    # Apply the embedding function on `x`
                    comput = self._get_embed_fn(element, verbose=verbose)
                    y = comput(x)
                    embed = tf.squeeze(y, axis=squeeze_axis, name='atomic')
                    if verbose:
                        log_tensor(embed)
                    values.append(embed)
            return tf.concat(values, axis=split_axis)

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
        atomic : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the energies
            of the real atoms.

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
            return tf.concat(
                [tf.add_n(stacks[el], name=el) for el in self._elements],
                axis=1, name='sum')

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
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            clf = self.transformer

            partitions, max_occurs = dynamic_partition(
                dists_and_masks=descriptors["radial"],
                elements=clf.elements,
                kbody_terms_for_element=clf.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=False,
            )

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

            partitions, max_occurs = dynamic_partition(
                dists_and_masks=descriptors["radial"],
                elements=clf.elements,
                kbody_terms_for_element=clf.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=True,
            )

            phi, _ = self._build_phi_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=verbose)

            y = tf.add(phi, embed, name='atomic')

            return y

    def export_to_setfl(self, setfl: str, nr: int, dr: float, nrho: int,
                        drho: float, r0=1.0, rt=None, rho0=0.0, rhot=None,
                        checkpoint=None, lattice_constants=None,
                        lattice_types=None, use_ema_variables=True):
        """
        Export this model to an `eam/alloy` or an `eam/fs` LAMMPS setfl
        potential file.

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
        checkpoint : str or None
            The tensorflow checkpoint file to restore. If None, the default
            (or initital) parameters will be used.
        lattice_constants : Dict[str, float] or None
            The lattice constant for each type of element.
        lattice_types : Dict[str, str] or None
            The lattice type, e.g 'fcc', for each type of element.
        use_ema_variables : bool
            If True, exponentially moving averaged variables will be used.

        """
        raise NotImplementedError(
            "This method must be overridden by its subclass")
