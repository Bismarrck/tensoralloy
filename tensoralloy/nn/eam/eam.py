# coding=utf-8
"""
This module defines the basic EAM-NN model.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from collections import Counter
from functools import partial
from typing import List, Dict, Callable
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.utils import get_elements_from_kbody_term, AttributeDict
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.eam.potentials import available_potentials
from tensoralloy.nn.eam.potentials import EamFSPotential, EamAlloyPotential

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def plot_potential(nx: int, dx: float, func: Callable, filename: str,
                   xlabel=None, ylabel=None, title=None):
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
    xlabel : str
        The label of X axis.
    ylabel : str
        The label of Y axis.
    title : str
        The title of the figure.

    """
    fig = plt.figure(1, figsize=[6, 6])

    x = np.arange(0.0, nx * dx, dx)
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

        super(EamNN, self).__init__(
            elements=elements,
            hidden_sizes=hidden_sizes,
            activation=activation,
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
                "activation": self._activation,
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
        name = self._potentials[element]['embed']
        if name == 'nn':
            return self._get_nn_fn(
                section=element,
                key='embed',
                variable_scope=f"{variable_scope}/{element}",
                verbose=verbose)
        else:
            return partial(self._empirical_functions[name].embed,
                           element=element,
                           variable_scope=variable_scope,
                           verbose=verbose)

    def _get_rho_fn(self, element_or_kbody_term: str, variable_scope='Rho',
                    verbose=False):
        """
        Return the electron density function of `name` for the given k-body
        term.
        """
        name = self._potentials[element_or_kbody_term]['rho']
        if name == 'nn':
            return self._get_nn_fn(
                section=element_or_kbody_term,
                key='rho',
                variable_scope=f"{variable_scope}/{element_or_kbody_term}",
                verbose=verbose)
        else:
            pot = self._empirical_functions[name]
            if isinstance(pot, EamAlloyPotential):
                return partial(pot.rho,
                               element=element_or_kbody_term,
                               variable_scope=variable_scope,
                               verbose=verbose)
            elif isinstance(pot, EamFSPotential):
                return partial(pot.rho,
                               kbody_term=element_or_kbody_term,
                               variable_scope=variable_scope,
                               verbose=verbose)
            else:
                raise ValueError(
                    f"Unknown EAM potential: {pot.__class__.__name__}")

    def _get_phi_fn(self, kbody_term: str, variable_scope='Phi', verbose=False):
        """
        Return the pairwise potential function of `name` for the given k-body
        term.
        """
        name = self._potentials[kbody_term]['phi']
        if name == 'nn':
            return self._get_nn_fn(
                section=kbody_term,
                key='phi',
                variable_scope=f"{variable_scope}/{kbody_term}",
                verbose=verbose)
        else:
            return partial(self._empirical_functions[name].phi,
                           kbody_term=kbody_term,
                           variable_scope=variable_scope,
                           verbose=verbose)

    def _get_internal_energy_op(self, outputs: tf.Tensor, features: AttributeDict,
                                name='energy', verbose=True):
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.
        features : AttributeDict
            A dict of input features.
        name : str
            The name of the output tensor.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        y_atomic = tf.identity(outputs, name='y_atomic')
        ndims = features.mask.shape.ndims
        axis = ndims - 1
        with tf.name_scope("Mask"):
            mask = tf.split(
                features.mask, [1, -1], axis=axis, name='split')[1]
            y_mask = tf.multiply(y_atomic, mask, name='mask')
        energy = tf.reduce_sum(
            y_mask, axis=axis, keepdims=False, name=name)
        if verbose:
            log_tensor(energy)
        return energy

    def _build_phi_nn(self, partitions: AttributeDict, max_occurs: Counter,
                      mode: tf_estimator.ModeKeys, verbose=False):
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
                    raw_values, masks = descriptors[element]
                    raw_values = tf.convert_to_tensor(raw_values, name='values')
                    masks = tf.convert_to_tensor(masks, name='masks')

                    if mode == tf_estimator.ModeKeys.PREDICT:
                        assert raw_values.shape.ndims == 4
                        values = tf.expand_dims(raw_values[0], axis=0)
                        masks = tf.expand_dims(masks, axis=0)
                        max_occurs[element] = tf.shape(values)[2]
                    else:
                        assert raw_values.shape.ndims == 5
                        values = raw_values[0]
                        max_occurs[element] = values.shape[2].value

                    num = len(kbody_terms)
                    glists = tf.split(
                        values, num_or_size_splits=num, axis=1, name='glist')
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

            y = tf.add(phi, embed, name='atomic')

            return y

    def export_to_setfl(self, setfl: str, nr: int, dr: float, nrho: int,
                        drho: float, checkpoint=None, lattice_constants=None,
                        lattice_types=None):
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
