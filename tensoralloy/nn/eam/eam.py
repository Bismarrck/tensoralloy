# coding=utf-8
"""
This module defines the basic EAM-NN model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from collections import Counter
from functools import partial
from typing import List, Dict

from tensoralloy.misc import AttributeDict
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.nn.eam.potentials import EamFSPotential, EamAlloyPotential

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class EamNN(BasicNN):
    """
    The tensorflow/CNN based implementation of the Embedded-Atom Method.
    """

    def __init__(self, custom_potentials=None, *args, **kwargs):
        """
        Initialization method.
        """
        self._unique_kbody_terms = None
        self._kbody_terms = None

        super(EamNN, self).__init__(*args, **kwargs)

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

    def _get_nn_fn(self, section, key, verbose=False):
        """
        Return a layer function of `f(x)` where `f` is a 1x1 CNN.
        """
        activation_fn = get_activation_fn(self._activation)
        hidden_sizes = self._hidden_sizes[section][key]
        return partial(self._get_1x1conv_nn, activation_fn=activation_fn,
                       hidden_sizes=hidden_sizes, verbose=verbose)

    def _get_embed_fn(self, element: str, verbose=False):
        """
        Return the embedding function of `name` for `element`.
        """
        name = self._potentials[element]['embed']
        if name == 'nn':
            return self._get_nn_fn(element, 'embed', verbose=verbose)
        else:
            return partial(self._empirical_functions[name].embed,
                           element=element)

    def _get_rho_fn(self, element_or_kbody_term: str, verbose=False):
        """
        Return the electron density function of `name` for the given k-body
        term.
        """
        name = self._potentials[element_or_kbody_term]['rho']
        if name == 'nn':
            return self._get_nn_fn(
                element_or_kbody_term, 'rho', verbose=verbose)
        else:
            pot = self._empirical_functions[name]
            if isinstance(pot, EamAlloyPotential):
                return partial(pot.rho, element=element_or_kbody_term)
            elif isinstance(pot, EamFSPotential):
                return partial(pot.rho, kbody_term=element_or_kbody_term)
            else:
                raise ValueError(
                    f"Unknown EAM potential: {pot.__class__.__name__}")

    def _get_phi_fn(self, kbody_term: str, verbose=False):
        """
        Return the pairwise potential function of `name` for the given k-body
        term.
        """
        name = self._potentials[kbody_term]['phi']
        if name == 'nn':
            return self._get_nn_fn(kbody_term, 'phi', verbose=verbose)
        else:
            return partial(self._empirical_functions[name].phi,
                           kbody_term=kbody_term)

    def _get_energy(self, outputs: List[tf.Tensor], features: AttributeDict,
                    verbose=True):
        """
        Return the Op to compute total energy of nn-EAM.

        Parameters
        ----------
        outputs : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.
        features : AttributeDict
            A dict of input features.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        with tf.name_scope("Energy"):
            y_atomic = tf.identity(outputs, name='atomic')
            shape = y_atomic.shape
            with tf.name_scope("mask"):
                mask = tf.split(
                    features.mask, [1, -1], axis=1, name='split')[1]
                y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_mask.set_shape(shape)
            energy = tf.reduce_sum(
                y_mask, axis=1, keepdims=False, name='energy')
            if verbose:
                log_tensor(energy)
            return energy

    def _build_phi_nn(self, partitions: AttributeDict, verbose=False):
        """
        Return the outputs of the pairwise interactions, `Phi(r)`.

        Parameters
        ----------
        partitions : AttributeDict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict. The keys are unique kbody terms and values are tuples of
            (value, mask) where `value` represents the descriptors and `mask` is
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, 1 + delta, max_n_element, nnl]`. `delta` will be zero
            if the corresponding kbody term has only one type of atom; otherwise
            `delta` will be one.
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
        with tf.name_scope("Phi") as scope:
            half = tf.constant(0.5, dtype=tf.float64, name='half')
            for kbody_term, (value, mask) in partitions.items():
                with tf.variable_scope(f"{scope}/{kbody_term}"):
                    # Convert `x` to a 5D tensor.
                    x = tf.expand_dims(value, axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    # Apply the `phi` function on `x`
                    comput = self._get_phi_fn(kbody_term, verbose=verbose)
                    y = comput(x)
                    # Apply the value mask
                    y = tf.multiply(y, tf.expand_dims(mask, axis=-1),
                                    name='masked')

                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    values[kbody_term] = y
                    y = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                    y = tf.squeeze(y, axis=1)
                    y = tf.multiply(y, half, name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs[kbody_term] = y
            return self._dynamic_stitch(outputs, symmetric=True), values

    def _build_rho_nn(self, values: AttributeDict, verbose=False):
        """
        Return the outputs of the electron densities, `rho(r)`.
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass")

    def _build_embed_nn(self, rho: tf.Tensor, max_occurs: Counter,
                        verbose=True):
        """
        Return the embedding energy, `F(rho)`.

        Parameters
        ----------
        rho : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the electron
            density of each atom.
        max_occurs : Counter
            The maximum occurance of each type of element.
        verbose : bool
            If True, key tensors will be logged.

        Returns
        -------
        embed : tf.Tensor
            The embedding energy. Has the same shape with `rho`.

        """
        split_sizes = [max_occurs[el] for el in self._elements]

        with tf.name_scope("Embed") as scope:
            splits = tf.split(rho, num_or_size_splits=split_sizes, axis=1)
            values = []
            for i, element in enumerate(self._elements):
                with tf.variable_scope(f"{scope}/{element}"):
                    x = tf.expand_dims(splits[i], axis=-1, name=element)
                    if verbose:
                        log_tensor(x)
                    # Apply the embedding function on `x`
                    comput = self._get_embed_fn(element, verbose=verbose)
                    y = comput(x)
                    embed = tf.squeeze(y, axis=2, name='atomic')
                    if verbose:
                        log_tensor(embed)
                    values.append(embed)
            return tf.concat(values, axis=1)

    def _dynamic_partition(self, descriptors: AttributeDict,
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
            the value mask. Both `value` and `mask` are 4D tensors of shape
            `[batch_size, max_n_terms, max_n_element, nnl]`.
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

        with tf.name_scope("Partition"):
            for element in self._elements:
                kbody_terms = self._kbody_terms[element]
                values, masks = descriptors.descriptors[element]
                max_occurs[element] = values.shape[2].value
                num = len(kbody_terms)
                glists = tf.split(values, num_or_size_splits=num, axis=1)
                mlists = tf.split(masks, num_or_size_splits=num, axis=1)
                for i, (value, mask) in enumerate(zip(glists, mlists)):
                    kbody_term = kbody_terms[i]
                    if merge_symmetric:
                        kbody_term = ''.join(
                            sorted(get_elements_from_kbody_term(kbody_term)))
                        if kbody_term in partitions:
                            value = tf.concat(
                                (partitions[kbody_term][0], value), axis=2)
                            mask = tf.concat(
                                (partitions[kbody_term][1], mask), axis=2)
                    partitions[kbody_term] = (value, mask)
            return partitions, Counter(max_occurs)

    def _dynamic_stitch(self, outputs: Dict[str, tf.Tensor], symmetric=False):
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
        symmetric : bool
            This should be True if symmetric tensors were splitted before.

        Returns
        -------
        atomic : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the energies
            of the real atoms.

        """
        with tf.name_scope("Stitch"):
            stacks = {}
            results = []
            for kbody_term, value in outputs.items():
                elements = get_elements_from_kbody_term(kbody_term)
                if symmetric and elements[0] != elements[1]:
                    results.append(value)
                else:
                    center = elements[0]
                    if center not in stacks:
                        stacks[center] = [value]
                    else:
                        stacks[center].append(value)
            results.append(
                tf.concat([tf.add_n(stacks[el], name=el)
                           for el in self._elements], axis=1))
            if not symmetric:
                return tf.identity(results[0], name='sum')
            else:
                return tf.add_n(results, name='sum')

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Return an nn-EAM model.
        """
        raise NotImplementedError(
            "This method must be overridden by its subclass")

    def export(self, setfl: str, nr: int, dr: float, nrho: int, drho: float,
               checkpoint=None, lattice_constants=None, lattice_types=None):
        """
        Export this model to an `eam/alloy` or an `eam/fs` potential file.

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
        raise NotImplementedError(
            "This method must be overridden by its subclass")
