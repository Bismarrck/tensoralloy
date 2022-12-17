#!coding=utf-8
"""
A special module for modeling finite-temperature systems.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from typing import List, Dict, Union
from collections import Counter

from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.utils import GraphKeys, ModeKeys, Defaults
from tensoralloy.nn import losses as loss_ops
from tensoralloy.nn.atomic.atomic import AtomicNN, Descriptor
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.dataclasses import EnergyOps, FiniteTemperatureEnergyOps
from tensoralloy.nn.dataclasses import EnergyOp
from tensoralloy.nn.dataclasses import LossParameters
from tensoralloy.nn.atomic.dataclasses import FiniteTemperatureOptions


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TemperatureDependentAtomicNN(AtomicNN):
    """
    Temperature-dependent atomistic neural network potential.
    """

    default_collection = GraphKeys.ATOMIC_NN_VARIABLES
    scope = "TD"

    def __init__(self, elements: List[str],
                 descriptor: Union[Descriptor, dict],
                 hidden_sizes=None,
                 activation=None,
                 kernel_initializer='he_normal',
                 minmax_scale=True,
                 use_resnet_dt=False,
                 atomic_static_energy=None,
                 use_atomic_static_energy=True,
                 fixed_atomic_static_energy=False,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces'),
                 finite_temperature=FiniteTemperatureOptions()):
        """
        Initialization method.
        """
        super(TemperatureDependentAtomicNN, self).__init__(
            elements=elements, descriptor=descriptor, hidden_sizes=hidden_sizes,
            activation=activation, kernel_initializer=kernel_initializer,
            minmax_scale=minmax_scale, use_resnet_dt=use_resnet_dt,
            atomic_static_energy=atomic_static_energy,
            fixed_atomic_static_energy=fixed_atomic_static_energy,
            use_atomic_static_energy=use_atomic_static_energy,
            minimize_properties=minimize_properties,
            export_properties=export_properties
        )

        if isinstance(finite_temperature, FiniteTemperatureOptions):
            self._finite_temperature = finite_temperature
        else:
            assert isinstance(finite_temperature, dict)
            self._finite_temperature = FiniteTemperatureOptions(
                **finite_temperature)

    @property
    def finite_temperature_options(self):
        """
        Return the options for modeling finite temperature systems.
        """
        return self._finite_temperature

    @property
    def is_finite_temperature(self) -> bool:
        """ Override this method. """
        return True

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        d = super(TemperatureDependentAtomicNN, self).as_dict()
        d['finite_temperature'] = self._finite_temperature.__dict__
        return d

    @staticmethod
    def _add_electron_temperature(x: tf.Tensor,
                                  etemperature: tf.Tensor,
                                  element: str,
                                  mode: ModeKeys,
                                  max_occurs: Counter):
        """
        Add electron temperature to the atomic descriptor tensor `x`.
        """
        with tf.name_scope("Temperature"):
            if ModeKeys.for_prediction(mode):
                np_dtype = x.dtype.as_numpy_dtype
                d0 = 1
                d2 = x.shape.dims[2].value
                vec = np.insert(np.zeros(d2), d2, 1).astype(np_dtype)
                mat = np.insert(np.eye(d2), d2, 0, axis=1).astype(np_dtype)
                vec = tf.convert_to_tensor(vec, name="vec", dtype=x.dtype)
                mat = tf.convert_to_tensor(mat, name="mat", dtype=x.dtype)
                x = tf.add(tf.einsum("nab,bc->nac", x, mat), 
                           vec * etemperature, name="x")
                etemp = tf.reshape(x[:, :, -1], (d0, -1, 1), name="etemp/tiled")
            else:
                d0, d1 = x.shape.as_list()[0: 2]
                etemp = tf.reshape(
                    etemperature, [d0, 1, 1], name='etemp')
                etemp = tf.tile(etemp, [1, d1, 1], name='etemp/tiled')
                x = tf.concat((x, etemp), axis=2, name='x')
        return x, etemp

    def _get_electron_entropy(self,
                              h: tf.Tensor,
                              t: tf.Tensor,
                              element: str,
                              collections: List[str],
                              verbose=True):
        """
        Model electron entropy S with the free electron model.

        Parameters
        ----------
        h : tf.Tensor
            Input features.
        t : tf.Tensor
            The electron temperature tensor.
        element : str
            The target element.
        collections : List[str]
            A list of str as the collections where the variables should be
            added.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        with tf.variable_scope("S"):
            eentropy = convolution1x1(
                h,
                activation_fn=get_activation_fn(self._activation),
                hidden_sizes=self._hidden_sizes[element],
                num_out=1,
                l2_weight=1.0,
                collections=collections,
                output_bias=True,
                output_bias_mean=0.0,
                use_resnet_dt=self._use_resnet_dt,
                kernel_initializer=self._kernel_initializer,
                variable_scope=None,
                verbose=verbose)
            eentropy = tf.squeeze(eentropy, axis=2, name="atomic")
            if verbose:
                log_tensor(eentropy)
            return eentropy

    def _get_internal_energy_outputs(self,
                                     h: tf.Tensor,
                                     element: str,
                                     atomic_static_energy: float,
                                     collections: List[str],
                                     verbose=True):
        """
        Model internal energy U using the temperature-dependent atomic
        descriptor 'h'.

        Parameters
        ----------
        h : tf.Tensor
            Input features.
        element : str
            The target element.
        atomic_static_energy : float
            Atomic static energy, used as the bias unit of the output layer.
        collections : List[str]
            A list of str as the collections where the variables should be
            added.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        with tf.variable_scope("U"):
            energy = convolution1x1(
                h,
                activation_fn=get_activation_fn(self._activation),
                hidden_sizes=self._hidden_sizes[element],
                num_out=1,
                l2_weight=1.0,
                collections=collections,
                output_bias=True,
                output_bias_mean=atomic_static_energy,
                use_resnet_dt=self._use_resnet_dt,
                kernel_initializer=self._kernel_initializer,
                variable_scope=None,
                verbose=verbose)
            energy = tf.squeeze(energy, axis=2, name="atomic")
            if verbose:
                log_tensor(energy)
            return energy

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
                           mode: ModeKeys,
                           verbose=False):
        """
        Build 1x1 Convolution1D based atomic neural networks for all elements.

        Parameters
        ----------
        features : dict
            A dict of input tensors and the descriptors:
                * 'descriptors'
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'atom_masks' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        descriptors : dict
            A dict of (element, (value, mask)) where `element` represents the
            symbol of an element, `value` is the descriptors of `element` and
            `mask` is None.
        mode : ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        collections = [self.default_collection]

        with tf.variable_scope(self.scope):

            outputs = {'energy': [], 'eentropy': [], 'free_energy': []}
            atomic_descriptors = self._descriptor.calculate(
                transformer=self._transformer,
                universal_descriptors=descriptors,
                mode=mode)
            for element, x in atomic_descriptors.descriptors.items():
                with tf.variable_scope(element, reuse=tf.AUTO_REUSE):
                    if self._use_atomic_static_energy:
                        bias_mean = self._atomic_static_energy.get(element, 0.0)
                    else:
                        bias_mean = 0.0
                    if verbose:
                        log_tensor(x)
                    if self._minmax_scale:
                        x = self._apply_minmax_normalization(
                            x=x,
                            mask=descriptors['atom_masks'][element],
                            mode=mode)
                        if verbose:
                            log_tensor(x)
                    activation_fn = get_activation_fn(
                        self._finite_temperature.activation)
                    layers = self._finite_temperature.layers
                    H = convolution1x1(
                        x,
                        activation_fn=activation_fn,
                        hidden_sizes=layers[:-1],
                        num_out=layers[-1],
                        l2_weight=1.0,
                        collections=collections,
                        kernel_initializer=self._kernel_initializer,
                        output_bias=True,
                        output_bias_mean=0.0,
                        use_resnet_dt=self._use_resnet_dt,
                        variable_scope="H",
                        verbose=verbose)
                    Ht, t = self._add_electron_temperature(
                        x=H,
                        etemperature=features["etemperature"],
                        element=element,
                        mode=mode,
                        max_occurs=atomic_descriptors.max_occurs)
                    T = tf.squeeze(t, axis=2, name='T')
                    S = self._get_electron_entropy(
                        h=Ht,
                        t=T,
                        element=element,
                        collections=collections,
                        verbose=verbose)
                    U = self._get_internal_energy_outputs(
                        h=Ht,
                        element=element,
                        atomic_static_energy=bias_mean,
                        collections=collections,
                        verbose=verbose)
                    U = tf.identity(U, name='U')
                    TS = tf.multiply(T, S, name='TS')
                    E = tf.subtract(U, TS, name='F')
                    outputs['energy'].append(U)
                    outputs['eentropy'].append(S)
                    outputs['free_energy'].append(E)
            return outputs

    def _get_energy_ops(self, outputs, features, verbose=True) -> EnergyOps:
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : Dict[str, [tf.Tensor]]
            A list of `tf.Tensor` as the outputs of the ANNs.
        features : dict
            A dict of input tensors.
        name : str
            The name of the output tensor.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        ops : EnergyOps
            The energy tensors.

        """
        ndims = features["atom_masks"].shape.ndims
        axis = ndims - 1
        with tf.name_scope("Mask"):
            mask = tf.split(
                features["atom_masks"], [1, -1], axis=axis, name='split')[1]

        def _build_energy_op(name: str) -> EnergyOp:
            name_map = {
                'energy': 'U',
                'eentropy': 'S',
                'free_energy': 'E'
            }
            with tf.name_scope(name_map[name]):
                eatom = tf.concat(outputs[name], axis=1, name='atomic/raw')
                if ndims == 1:
                    eatom = tf.squeeze(eatom, axis=0)
                eatom = tf.multiply(eatom, mask, name='atomic')
            etotal = tf.reduce_sum(
                eatom, axis=axis, keepdims=False, name=name)
            return EnergyOp(total=etotal, atomic=eatom)

        free_energy = _build_energy_op('free_energy')
        eentropy = _build_energy_op('eentropy')
        energy = _build_energy_op('energy')
        if verbose:
            log_tensor(free_energy.total)
            log_tensor(eentropy.total)
            log_tensor(energy.total)
        return FiniteTemperatureEnergyOps(
            energy=energy, eentropy=eentropy, free_energy=free_energy)

    def _get_energy_loss(self,
                         predictions,
                         labels,
                         n_atoms,
                         max_train_steps,
                         loss_parameters: LossParameters,
                         collections) -> Dict[str, tf.Tensor]:
        """
        The energy loss or energy losses if temperature effect is considered.
        """

        losses = {}
        with tf.name_scope("Energy"):
            for scope_name, prop in {"U": "energy",
                                     "F": "free_energy",
                                     "S": "eentropy"}.items():
                if prop in self._minimize_properties:
                    loss = loss_ops.get_energy_loss(
                        labels=labels[prop],
                        predictions=predictions[prop],
                        n_atoms=n_atoms,
                        max_train_steps=max_train_steps,
                        options=loss_parameters[prop],
                        collections=collections,
                        name_scope=scope_name)
                    losses[prop] = loss
        return losses
        
    def export_to_lammps_native(self, model_path, checkpoint=None, 
                                use_ema_variables=True, dtype=np.float64):
        """
        Export the model for LAMMPS pair_style tensoralloy/native.
        """
        from tensoralloy.nn.atomic.grap import GenericRadialAtomicPotential
        from ase.data import atomic_masses, atomic_numbers

        if not isinstance(self._descriptor, GenericRadialAtomicPotential):
            raise ValueError(
                "The descriptor GenericRadialAtomicPotential is required")
        if self._descriptor.algorithm.name not in ("pexp", "nn"):
            raise ValueError("Only (pexp, nn) are supported!")
        
        layer_sizes = np.array(self._hidden_sizes[self._elements[0]], dtype=int)
        for elt in self._elements[1:]:
            if not np.all(self._hidden_sizes[elt] == layer_sizes):
                raise ValueError("Layer sizes of all elements must be the same")
        layer_sizes = np.append(layer_sizes, 1).astype(np.int32)

        fctype_map = {"cosine": 0, "polynomial": 1}
        actfn_map = {"relu": 0, "softplus": 1, "tanh": 2}
        
        graph = tf.Graph()
        with graph.as_default():
            if self._transformer is None:
                raise ValueError("A transformer must be attached before "
                                 "exporting to a pb file.")
            configs = self.as_dict()
            configs.pop('class')
            nn = self.__class__(**configs)

            if isinstance(self._transformer, BatchUniversalTransformer):
                clf = self._transformer.as_descriptor_transformer()
            else:
                serialized = self._transformer.as_dict()
                if 'class' in serialized:
                    serialized.pop('class')
                clf = self._transformer.__class__(**serialized)

            nn.attach_transformer(clf)
            nn.build(clf.get_placeholder_features(),
                     mode=ModeKeys.PREDICT,
                     verbose=True)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                if use_ema_variables:
                    # Restore the moving averaged variables
                    ema = tf.train.ExponentialMovingAverage(
                        Defaults.variable_moving_average_decay)
                    saver = tf.train.Saver(ema.variables_to_restore())
                else:
                    saver = tf.train.Saver(var_list=tf.model_variables())
                if checkpoint is not None:
                    saver.restore(sess, checkpoint)

                elements = clf.elements
                masses = [atomic_masses[atomic_numbers[elt]] 
                          for elt in elements]
                chars = []
                for elt in elements:
                    for char in elt:
                        chars.append(ord(char))
                
                data = {
                    "rmax": dtype(clf.rcut),
                    "nelt": np.int32(len(clf.elements)),
                    "masses": np.array(masses, dtype=dtype),
                    "numbers": np.array(chars, dtype=dtype)
                }
                data["max_moment"] = np.int32(self._descriptor.max_moment)
                data["fctype"] = np.int32(
                    fctype_map[self._descriptor.cutoff_function])
                data["tdnp"] = np.int32(1)
                data["precision"] = np.int32(64 if dtype == np.float64 else 32)
                data["is_T_symmetric"] = np.int32(
                    self._descriptor.is_T_symmetric)

                # --------------------------------------------------------------
                # Algorithm
                # 

                algo = self._descriptor.algorithm.as_dict(convert_to_pairs=True)
                if self._descriptor.algorithm.name == "pexp":
                    data["descriptor::method"] = np.int32(0)
                    data["descriptor::rl"] = np.array(
                        algo["parameters"]["rl"], dtype=dtype)
                    data["descriptor::pl"] = np.array(
                        algo["parameters"]["pl"], dtype=dtype)
                elif self._descriptor.algorithm.name == "morse":
                    data["descriptor::method"] = np.int32(1)
                    data["descriptor::D"] = np.array(
                        algo["parameters"]["D"], dtype=dtype)
                    data["descriptor::gamma"] = np.array(
                        algo["parameters"]["gamma"], dtype=dtype)
                    data["descriptor::r0"] = np.array(
                        algo["parameters"]["r0"], dtype=dtype)
                elif self._descriptor.algorithm.name == "density":
                    data["descriptor::method"] = np.int32(2)
                    data["descriptor::A"] = np.array(
                        algo["parameters"]["A"], dtype=dtype)
                    data["descriptor::beta"] = np.array(
                        algo["parameters"]["beta"], dtype=dtype)
                    data["descriptor::re"] = np.array(
                        algo["parameters"]["re"], dtype=dtype)
                elif self._descriptor.algorithm.name == "sf":
                    data["descriptor::method"] = np.int32(3)
                    data["descriptor::eta"] = np.array(
                        algo["parameters"]["eta"], dtype=dtype)
                    data["descriptor::omega"] = np.array(
                        algo["parameters"]["omega"], dtype=dtype)
                else:
                    data["use_fnn"] = np.int32(1)
                    data["fnn::nlayers"] = np.int32(
                        len(algo["hidden_sizes"]) + 1)
                    data["fnn::layer_sizes"] = np.array(
                        np.append(algo["hidden_sizes"], algo["num_filters"]),
                        dtype=np.int32)
                    data["fnn::num_filters"] = np.int32(algo["num_filters"])
                    data["fnn::actfn"] = np.int32(actfn_map[algo["activation"]])
                    data["fnn::use_resnet_dt"] = np.int32(algo["use_resnet_dt"])
                    data["fnn::apply_output_bias"] = np.int32(0)
                    for j in range(data["fnn::nlayers"] - 1):
                        ops = [
                            graph.get_tensor_by_name(
                                f"{self.scope}/Filters/Conv3d{j + 1}/kernel:0"),
                            graph.get_tensor_by_name(
                                f"{self.scope}/Filters/Conv3d{j + 1}/bias:0")
                        ]
                        weights, biases = sess.run(ops)
                        weights = np.squeeze(weights).astype(dtype)
                        biases = np.squeeze(biases).astype(dtype)
                        data[f"fnn::weights_0_{j}"] = weights
                        data[f"fnn::biases_0_{j}"] = biases
                    ops = [
                        graph.get_tensor_by_name(
                            f"{self.scope}/Filters/Output/kernel:0"),
                    ]
                    weights = np.squeeze(sess.run(ops)[0]).astype(dtype)
                    data[f"fnn::weights_0_{data['fnn::nlayers'] - 1}"] = weights

                # --------------------------------------------------------------
                # H
                # 

                nlayers = len(self._finite_temperature.layers)
                data["H::nlayers"] = np.int32(nlayers)
                data["H::actfn"] = np.int32(
                    actfn_map[self._finite_temperature.activation])
                data["H::layer_sizes"] = np.array(
                    self._finite_temperature.layers, dtype=np.int32)
                data["H::use_resnet_dt"] = np.int32(self._use_resnet_dt)
                data["H::apply_output_bias"] = np.int32(1)
                for i, elt in enumerate(elements):
                    for j in range(nlayers - 1):
                        ops = [
                            graph.get_tensor_by_name(
                                f"{self.scope}/{elt}/H/Conv1d{j + 1}/kernel:0"),
                            graph.get_tensor_by_name(
                                f"{self.scope}/{elt}/H/Conv1d{j + 1}/bias:0")
                        ]
                        weights, biases = sess.run(ops)
                        weights = np.squeeze(weights).astype(dtype)
                        biases = np.squeeze(biases).astype(dtype)
                        data[f"H::weights_{i}_{j}"] = weights
                        data[f"H::biases_{i}_{j}"] = biases
                    ops = [
                        graph.get_tensor_by_name(
                            f"{self.scope}/{elt}/H/Output/kernel:0"),
                        graph.get_tensor_by_name(
                            f"{self.scope}/{elt}/H/Output/bias:0")
                    ]
                    results = sess.run(ops)
                    weights = np.squeeze(results[0]).astype(dtype)
                    biases = np.squeeze(results[1]).astype(dtype)
                    data[f"H::weights_{i}_{nlayers - 1}"] = weights
                    data[f"H::biases_{i}_{nlayers - 1}"] = biases
                
                # --------------------------------------------------------------
                # S
                # 

                nlayers = len(layer_sizes)
                data["S::nlayers"] = np.int32(nlayers)
                data["S::actfn"] = np.int32(actfn_map[self._activation])
                data["S::layer_sizes"] = np.array(layer_sizes, dtype=np.int32)
                data["S::use_resnet_dt"] = np.int32(self._use_resnet_dt)
                data["S::apply_output_bias"] = np.int32(1)
                for i, elt in enumerate(elements):
                    for j in range(nlayers - 1):
                        ops = [
                            graph.get_tensor_by_name(
                                f"{self.scope}/{elt}/S/Conv1d{j + 1}/kernel:0"),
                            graph.get_tensor_by_name(
                                f"{self.scope}/{elt}/S/Conv1d{j + 1}/bias:0")
                        ]
                        weights, biases = sess.run(ops)
                        weights = np.squeeze(weights).astype(dtype)
                        biases = np.squeeze(biases).astype(dtype)
                        data[f"S::weights_{i}_{j}"] = weights
                        data[f"S::biases_{i}_{j}"] = biases
                    ops = [
                        graph.get_tensor_by_name(
                            f"{self.scope}/{elt}/S/Output/kernel:0"),
                        graph.get_tensor_by_name(
                            f"{self.scope}/{elt}/S/Output/bias:0")
                    ]
                    results = sess.run(ops)
                    weights = np.squeeze(results[0]).astype(dtype)
                    biases = np.squeeze(results[1]).astype(dtype)
                    data[f"S::weights_{i}_{nlayers - 1}"] = weights
                    data[f"S::biases_{i}_{nlayers - 1}"] = biases
                
                # --------------------------------------------------------------
                # U
                # 

                nlayers = len(layer_sizes)
                data["U::nlayers"] = np.int32(nlayers)
                data["U::actfn"] = np.int32(actfn_map[self._activation])
                data["U::layer_sizes"] = np.array(layer_sizes, dtype=np.int32)
                data["U::use_resnet_dt"] = np.int32(self._use_resnet_dt)
                data["U::apply_output_bias"] = np.int32(
                    self._use_atomic_static_energy)
                for i, elt in enumerate(elements):
                    for j in range(nlayers - 1):
                        ops = [
                            graph.get_tensor_by_name(
                                f"{self.scope}/{elt}/U/Conv1d{j + 1}/kernel:0"),
                            graph.get_tensor_by_name(
                                f"{self.scope}/{elt}/U/Conv1d{j + 1}/bias:0")
                        ]
                        weights, biases = sess.run(ops)
                        weights = np.squeeze(weights).astype(dtype)
                        biases = np.squeeze(biases).astype(dtype)
                        data[f"U::weights_{i}_{j}"] = weights
                        data[f"U::biases_{i}_{j}"] = biases
                    ops = [
                        graph.get_tensor_by_name(
                            f"{self.scope}/{elt}/U/Output/kernel:0"),
                    ]
                    if self._use_atomic_static_energy:
                        ops.append(
                            graph.get_tensor_by_name(
                                f"{self.scope}/{elt}/U/Output/bias:0"))
                    results = sess.run(ops)
                    weights = np.squeeze(results[0]).astype(dtype)
                    data[f"U::weights_{i}_{nlayers - 1}"] = weights
                    if len(results) == 2:
                        biases = np.squeeze(results[1]).astype(dtype)
                        data[f"U::biases_{i}_{nlayers - 1}"] = biases
                
                np.savez(model_path, **data)
