# coding=utf-8
"""
This module defines the basic neural network for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
import shutil
import os

from datetime import datetime
from typing import List, Dict
from os.path import join, dirname, exists
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.tensor_util import is_tensor
from tensorflow_estimator import estimator as tf_estimator
from ase.units import GPa

from tensoralloy.utils import GraphKeys, Defaults, safe_select
from tensoralloy.nn.dataclasses import StructuralProperty, LossParameters
from tensoralloy.nn.utils import log_tensor, is_first_replica
from tensoralloy.nn.opt import get_train_op, get_training_hooks
from tensoralloy.nn.eval import get_eval_metrics_ops, get_evaluation_hooks
from tensoralloy.nn import losses as loss_ops
from tensoralloy.nn.losses import LossMethod
from tensoralloy.nn.constraint import elastic as elastic_ops
from tensoralloy.nn.constraint import rose as rose_ops
from tensoralloy.transformer.base import BaseTransformer
from tensoralloy.transformer.base import BatchDescriptorTransformer
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.precision import get_float_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class _PropertyError(ValueError):
    """
    This error shall be raised if the given property is not valid.
    """
    label = "valid"

    def __init__(self, name):
        super(_PropertyError, self).__init__()
        self.name = name

    def __str__(self):
        return f"'{self.name}' is not a '{self.label}' property."


class MinimizablePropertyError(_PropertyError):
    """
    This error shall be raised if the given property cannot be minimized.
    """
    label = "minimizable"


class ExportablePropertyError(_PropertyError):
    """
    This error shall be raised if the given property cannot be exported.
    """
    label = "exportable"


all_properties = (
    StructuralProperty(name='energy'),
    StructuralProperty(name='atomic', exportable=True, minimizable=False),
    StructuralProperty(name='forces'),
    StructuralProperty(name="partial_forces", exportable=True,
                       minimizable=False),
    StructuralProperty(name='stress'),
    StructuralProperty(name='total_pressure'),
    StructuralProperty(name='hessian', minimizable=False),
    StructuralProperty(name='elastic'),
    StructuralProperty(name='rose', exportable=False)
)

exportable_properties = [
    prop for prop in all_properties if prop.exportable
]

minimizable_properties = [
    prop for prop in all_properties if prop.minimizable
]


class BasicNN:
    """
    The base neural network class.
    """

    # The default collection for model variabls.
    default_collection = None

    def __init__(self,
                 elements: List[str],
                 hidden_sizes=None,
                 activation=None,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces')):
        """
        Initialization method.

        Parameters
        ----------
        elements : List[str]
            A list of str as the ordered elements.
        hidden_sizes : int or List[int] or Dict[str, List[int]] or array_like
            A list of int or a dict of (str, list of int) or an int as the sizes
            of the hidden layers.
        activation : str
            The name of the activation function to use.
        minimize_properties : List[str]
            A list of str as the properties to minimize. Avaibale properties
            are: 'energy', 'forces', 'stress' and 'total_pressure'. For each
            property, its RMSE loss will be minimized.
        export_properties : List[str]
            A list of str as the properties to infer when exporting the model.
            'energy' will always be exported.

        Notes
        -----
        At least one of `energy`, `forces`, `stress` or `total_pressure` must be
        True.

        """
        self._elements = sorted(list(elements))
        self._hidden_sizes = self._get_hidden_sizes(
            safe_select(hidden_sizes, Defaults.hidden_sizes))
        self._activation = safe_select(activation, Defaults.activation)

        if len(minimize_properties) == 0:
            raise ValueError("At least one property should be minimized.")

        for prop in minimize_properties:
            if prop not in minimizable_properties:
                raise MinimizablePropertyError(prop)

        for prop in export_properties:
            if prop not in exportable_properties:
                raise ExportablePropertyError(prop)

        self._minimize_properties = list(minimize_properties)
        self._export_properties = list(export_properties)
        self._transformer = None
        self._y_atomic_op_name = None

    @property
    def elements(self):
        """
        Return the ordered elements.
        """
        return self._elements

    @property
    def hidden_sizes(self):
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    @property
    def minimize_properties(self) -> List[str]:
        """
        Return a list of str as the properties to minimize.
        """
        return self._minimize_properties

    @property
    def predict_properties(self) -> List[str]:
        """
        Return a list of str as the properties to predict.
        """
        return self._export_properties

    def _get_atomic_energy_tensor_name(self) -> str:
        """
        Return the name of the atomic energy tensor.
        """
        return self._y_atomic_op_name

    def _get_hidden_sizes(self, hidden_sizes):
        """
        Convert `hidden_sizes` to a dict if needed.
        """
        results = {}
        for element in self._elements:
            if isinstance(hidden_sizes, dict):
                sizes = np.asarray(
                    hidden_sizes.get(element, Defaults.hidden_sizes),
                    dtype=np.int)
            else:
                sizes = np.atleast_1d(hidden_sizes).astype(np.int)
            assert (sizes > 0).all()
            results[element] = sizes.tolist()
        return results

    @property
    def transformer(self) -> BaseTransformer:
        """
        Get the attached transformer.
        """
        return self._transformer

    def attach_transformer(self, clf: BaseTransformer):
        """
        Attach a descriptor transformer to this NN.
        """
        self._transformer = clf

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        raise NotImplementedError("This method must be overridden!")

    def _get_internal_energy_op(self,
                                outputs,
                                features: dict,
                                name='energy',
                                verbose=True) -> tf.Tensor:
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : Any
            The model outputs from the method `_get_model_outputs`.
        features : dict
            A dict of input raw property tensors:
                * 'positions' of shape `[batch_size, n_atoms_max + 1, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'atom_masks' of shape `[batch_size, n_atoms_max + 1]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms_vap' of dtype `int64`.
                * 'etemperature' of dtype `float32` or `float64`.
                * 'pulay_stress' of dtype `float32` or `float64`.
        name : str
            The name of the output potential energy tensor.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        raise NotImplementedError("This method must be overridden!")

    @staticmethod
    def _get_pv_energy_op(features: dict,
                          name='pv',
                          verbose=True) -> tf.Tensor:
        """
        Return the Op to compute `E_pv`:

            E(pv) = pulay_stress * volume

        """
        v = tf.linalg.det(features["cell"], name='V')
        p = tf.convert_to_tensor(features["pulay_stress"], name='P')
        pv = tf.multiply(v, p, name=name)
        if verbose:
            log_tensor(pv)
        return pv

    def _get_total_energy_op(self,
                             outputs,
                             features: dict,
                             name='energy',
                             verbose=True):
        """
        Return the Ops to compute total energy E and enthalpy H:

            H = E + PV

        Returns
        -------
        energy : tf.Tensor
            The Op to compute internal energy E.
        enthalpy : tf.Tensor
            The Op to compute enthalpy H.

        """
        with tf.name_scope("PV"):
            E_pv = self._get_pv_energy_op(features, verbose=verbose)

        with tf.name_scope("Internal"):
            E_internal = self._get_internal_energy_op(
                outputs=outputs,
                features=features,
                verbose=verbose)

        E_total = tf.identity(E_internal, name=name)
        H = tf.add(E_internal, E_pv, name='enthalpy')

        if verbose:
            log_tensor(H)

        return E_total, H

    @staticmethod
    def _get_forces_op(energy, positions, name='forces', verbose=True):
        """
        Return the Op to compute atomic forces (eV / Angstrom).
        """
        dEdR = tf.gradients(energy, positions, name='dEdR')[0]
        # Setup the splitting axis. `energy` may be a 1D vector or a scalar.
        axis = energy.shape.ndims
        # Please remember: f = -dE/dR
        forces = tf.negative(
            tf.split(dEdR, [1, -1], axis=axis, name='split')[1],
            name=name)
        if verbose:
            log_tensor(forces)
        return forces

    @staticmethod
    def _get_partial_forces_ops(energy, rij, rijk, verbose=True):
        """
        Return the Ops for computing partial forces.
        """
        ops = {}
        dEdrij = tf.gradients(energy, rij, name='dEdrij')[0]
        ops["dEdrij"] = tf.identity(dEdrij, name='dE/drij')
        if rijk is not None:
            dEdrijk = tf.gradients(energy, rijk, name='dEdrijk')[0]
            ops["dEdrijk"] = tf.identity(dEdrijk, name='dE/drijk')
        if verbose:
            for tensor in ops.values():
                log_tensor(tensor)
        return ops

    @staticmethod
    def _get_reduced_full_stress_tensor(energy: tf.Tensor, cell, volume,
                                        positions, forces, pulay_stress):
        """
        Return the Op to compute the virial stress tensor.

            (stress - pulay_stress) * volume = -F^T @ R + (dE/dh)^T @ h

        where `E` denotes the total energy, `h` is the 3x3 row-major lattice
        matrix, `R` and `F` are positions and total forces of the atoms.
        """
        with tf.name_scope("Full"):
            # The cell tensors in Python/ASE are row-major. So `dE/dh` must be
            # transposed.
            dEdh = tf.identity(tf.gradients(energy, cell)[0], name='dEdh')
            dtype = dEdh.dtype
            if cell.shape.ndims == 2:
                with tf.name_scope("Right"):
                    right = tf.matmul(tf.transpose(dEdh, name='dEdhT'), cell,
                                      name='right')
                with tf.name_scope("Left"):
                    positions = tf.split(
                        positions, [1, -1], axis=0, name='split')[1]
                    left = tf.matmul(tf.transpose(forces), positions)
                    left = tf.negative(left, name='left')
                internal = tf.add(left, right, name='internal')
                with tf.name_scope("PV"):
                    pv = tf.multiply(tf.eye(3, dtype=dtype),
                                     pulay_stress * volume,
                                     name='pv')
                total_stress = tf.subtract(internal, pv, name='stress')
                stress = tf.math.truediv(total_stress, volume, name='ase')
            else:
                with tf.name_scope("Right"):
                    right = tf.einsum('ikj,ikl->ijl', dEdh, cell, name='right')
                with tf.name_scope("Left"):
                    positions = tf.split(
                        positions, [1, -1], axis=1, name='split')[1]
                    left = tf.einsum('ijk,ijl->ijlk', positions, forces)
                    left = tf.reduce_sum(left, axis=1, keepdims=False)
                    left = tf.negative(left, name='left')
                internal = tf.add(left, right, name='internal')

                with tf.name_scope("PV"):
                    batch_shape = [energy.shape.as_list()[0]]
                    pv = tf.multiply(volume, pulay_stress, name='pv')
                    pv = tf.multiply(
                        tf.eye(3, batch_shape=batch_shape, dtype=dtype),
                        tf.reshape(pv, [-1, 1, 1]),
                        name='pv')
                total_stress = tf.subtract(internal, pv, name='stress')
                stress = tf.math.truediv(tf.reshape(total_stress, (-1, 9)),
                                         tf.reshape(volume, (-1, 1)))
                stress = tf.reshape(stress, (-1, 3, 3), name='ase')
            return tf.identity(stress, name='unit'), total_stress

    @staticmethod
    def _convert_to_voigt_stress(stress, batch_size, name='stress',
                                 verbose=False):
        """
        Convert a 3x3 stress tensor or a Nx3x3 stress tensors to corresponding
        Voigt form(s).
        """
        ndims = stress.shape.ndims
        with tf.name_scope("Voigt"):
            voigt = tf.convert_to_tensor(
                [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]],
                dtype=tf.int32, name='voigt')
            if ndims == 3:
                voigt = tf.tile(
                    tf.reshape(voigt, [1, 6, 2]), (batch_size, 1, 1))
                indices = tf.tile(tf.reshape(
                    tf.range(batch_size), [batch_size, 1, 1]),
                    [1, 6, 1])
                voigt = tf.concat((indices, voigt), axis=2, name='indices')
            stress = tf.gather_nd(stress, voigt, name=name)
            if verbose:
                log_tensor(stress)
            return stress

    def _get_stress_op(self, energy: tf.Tensor, cell, volume, positions,
                       forces, pulay_stress, name='stress',
                       return_pressure=False, verbose=True):
        """
        Return the Op to compute the reduced stress (eV) in Voigt format.
        """

        # Get the 3x3 full stress tensor
        stress_per_volume, total_stress = \
            self._get_reduced_full_stress_tensor(
                energy, cell, volume, positions, forces, pulay_stress)
        if verbose:
            log_tensor(total_stress)

        # Get the Voigt stress tensor
        ndims = stress_per_volume.shape.ndims
        batch_size = cell.shape[0].value or energy.shape[0].value
        if ndims == 3 and batch_size is None:
            raise ValueError("The batch size cannot be inferred.")
        voigt = self._convert_to_voigt_stress(
            stress_per_volume, batch_size, name=name, verbose=verbose)

        if return_pressure:
            total_pressure = self._get_total_pressure_op(stress_per_volume,
                                                         verbose=verbose)
        else:
            total_pressure = tf.no_op()

        return voigt, total_stress, total_pressure

    @staticmethod
    def _get_total_pressure_op(stress_per_volume: tf.Tensor, name='pressure',
                               verbose=True):
        """
        Return the Op to compute the reduced total pressure (eV).

            reduced_total_pressure = -trace(full_stress) / 3.0

        """
        dtype = stress_per_volume.dtype
        total_pressure = tf.math.truediv(tf.trace(stress_per_volume),
                                         tf.constant(-3.0 * GPa, dtype=dtype),
                                         name=f"{name}/GPa")
        if verbose:
            log_tensor(total_pressure)
        return total_pressure

    @staticmethod
    def _get_hessian_op(energy: tf.Tensor, positions: tf.Tensor, name='hessian',
                        verbose=True):
        """
        Return the Op to compute the Hessian matrix:

            hessian = d^2E / dR^2 = d(dE / dR) / dR

        """
        hessian = tf.identity(tf.hessians(energy, positions)[0], name=name)
        if verbose:
            log_tensor(hessian)
        return hessian

    def get_total_loss(self, predictions, labels, n_atoms, atom_masks,
                       loss_parameters: LossParameters,
                       mode=tf_estimator.ModeKeys.TRAIN):
        """
        Get the total loss tensor.

        Parameters
        ----------
        predictions : dict
            A dict of tensors as the predictions.
                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
                  required if 'forces' should be minimized.
                * 'stress' of shape `[batch_size, 6]` is required if
                  'stress' should be minimized. Its unit should be `eV/Ang**3`.
                * 'total_pressure' of shape `[batch_size, ]` is required if
                  'total_pressure' should be minimized.

            The following prediction is included and required for computing
            elastic loss:
                * 'total_stress' of shape `[batch_size, 3, 3]` with unit `eV`.
        labels : dict
            A dict of reference tensors.

            Always required:
                * 'energy' of shape `[batch_size, ]`.

            Required if 'forces' should be minimized:
                * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
                  required if 'forces' should be minimized.

            Required if 'stress' or 'total_pressure' should be minimized:
                * 'stress' of shape `[batch_size, 6]` is required if
                  'stress' should be minimized.
                * 'pulay_stress' of shape `[batch_size, ]`

        n_atoms : tf.Tensor
            A `int64` tensor of shape `[batch_size, ]`.
        atom_masks : tf.Tensor
            A float tensor of shape `[batch_size, n_atoms_max + 1]`.
        loss_parameters : LossParameters
            The hyper parameters for computing the total loss.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.

        Returns
        -------
        total_loss : tf.Tensor
            A `float64` tensor as the total loss.
        losses : dict
            A dict. The loss tensor for energy, forces and stress or total
            pressure.

        """
        with tf.name_scope("Loss"):

            if is_first_replica():
                collections = [GraphKeys.TRAIN_METRICS]
            else:
                collections = None

            losses = dict()
            losses["energy"] = loss_ops.get_energy_loss(
                labels=labels["energy"],
                predictions=predictions["energy"],
                n_atoms=n_atoms,
                loss_weight=loss_parameters.energy.weight,
                per_atom_loss=loss_parameters.energy.per_atom_loss,
                method=LossMethod[loss_parameters.energy.method],
                collections=collections)

            if 'forces' in self._minimize_properties:
                losses["forces"] = loss_ops.get_forces_loss(
                    labels=labels["forces"],
                    predictions=predictions["forces"],
                    atom_masks=atom_masks,
                    loss_weight=loss_parameters.forces.weight,
                    method=LossMethod[loss_parameters.forces.method],
                    collections=collections)

            if 'total_pressure' in self._minimize_properties:
                losses["total_pressure"] = loss_ops.get_total_pressure_loss(
                    labels=labels["total_pressure"],
                    predictions=predictions["total_pressure"],
                    loss_weight=loss_parameters.total_pressure.weight,
                    method=LossMethod[loss_parameters.total_pressure.method],
                    collections=collections)

            if 'stress' in self._minimize_properties:
                losses["stress"] = loss_ops.get_stress_loss(
                    labels=labels["stress"],
                    predictions=predictions["stress"],
                    loss_weight=loss_parameters.stress.weight,
                    method=LossMethod[loss_parameters.stress.method],
                    collections=collections)

            losses["l2"] = loss_ops.get_l2_regularization_loss(
                options=loss_parameters.l2,
                collections=collections)

            verbose = bool(mode == tf_estimator.ModeKeys.TRAIN)

            if 'elastic' in self._minimize_properties:
                if loss_parameters.elastic.crystals is not None:
                    losses["elastic"] = elastic_ops.get_elastic_constant_loss(
                        base_nn=self,
                        list_of_crystal=loss_parameters.elastic.crystals,
                        weight=loss_parameters.elastic.weight,
                        options=loss_parameters.elastic.constraint,
                        verbose=verbose)

            if 'rose' in self._minimize_properties:
                if loss_parameters.rose.crystals is not None:
                    losses["rose"] = rose_ops.get_rose_constraint_loss(
                        base_nn=self,
                        options=loss_parameters.rose,
                        verbose=verbose)

            for tensor in losses.values():
                tf.compat.v1.summary.scalar(tensor.op.name + '/summary', tensor)

        total_loss = tf.add_n(list(losses.values()), name='total_loss')

        if is_first_replica():
            tf.add_to_collection(GraphKeys.TRAIN_METRICS, total_loss)

        return total_loss, losses

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
                           mode: tf_estimator.ModeKeys,
                           verbose=False):
        """
        Build the NN model and return raw outputs.

        Parameters
        ----------
        features : dict
            A dict of input raw property tensors:
                * 'positions' of shape `[batch_size, n_atoms_max + 1, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'atom_masks' of shape `[batch_size, n_atoms_max + 1]`.
                * 'etemperature' of dtype `float32` or `float64`
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms_vap' of dtype `int64`.'
                * 'pulay_stress' of dtype `float32` or `float64`.
        descriptors : dict
            A dict of Ops to get atomic descriptors. This should be produced by
            an overrided `BaseTransformer.get_descriptors()`.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        raise NotImplementedError("This method must be overridden!")

    def _check_keys(self, features: dict, labels: dict):
        """
        Check the keys of `features` and `labels`.
        """
        assert 'positions' in features
        assert 'cell' in features
        assert 'atom_masks' in features
        assert 'n_atoms_vap' in features
        assert 'volume' in features
        assert isinstance(self._transformer, BaseTransformer)

        for prop in self._minimize_properties:
            if prop in ('elastic', 'rose'):
                continue
            assert prop in labels

    def build(self,
              features: dict,
              mode=tf_estimator.ModeKeys.TRAIN,
              verbose=True):
        """
        Build the atomic neural network.

        Parameters
        ----------
        features : dict
            A dict of input raw property tensors:
                * 'positions' of shape `[batch_size, n_atoms_max + 1, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'atom_masks' of shape `[batch_size, n_atoms_max + 1]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms_vap' of dtype `int64`.
                * 'etemperature' of dtype `float32` or `float64`
                * 'pulay_stress' of dtype `float32` or `float64`.
        mode : tf_estimator.ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        Returns
        -------
        predictions : dict
            A dict of tensors as the predictions. If `mode` is PREDICT, the
            first axes (batch size) are ignored.

                * 'energy' of shape `[batch_size, ]` is required.
                * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
                  required if 'forces' should be minimized.
                * 'stress' of shape `[batch_size, 6]` is required if
                  'stress' should be minimized. Its unit should be `eV/Ang**3`.
                * 'total_pressure' of shape `[batch_size, ]` is required if
                  'total_pressure' should be minimized.

            The following prediction is included and required for computing
            elastic constants:
                * 'total_stress' of shape `[batch_size, 3, 3]` with unit `eV`.

        """

        if self._transformer is None:
            raise ValueError("A descriptor transformer must be attached.")

        # 'descriptors', a dict of (element, (value, mask)) where `element`
        # represents the symbol of an element, `value` is the descriptors of
        # `element` and `mask` is the mask of `value`.
        descriptors = self._transformer.get_descriptors(features)

        outputs = self._get_model_outputs(
            features=features,
            descriptors=descriptors,
            mode=mode,
            verbose=verbose)

        export_partial_forces = False
        if mode == tf_estimator.ModeKeys.PREDICT:
            if isinstance(self._transformer, UniversalTransformer) \
                    and not self._transformer.use_computed_dists:
                export_partial_forces = True
                properties = ["energy", "partial_forces"]
            else:
                properties = self._export_properties
        else:
            properties = self._minimize_properties

        with tf.name_scope("Output"):

            predictions = dict()

            with tf.name_scope("Energy"):
                energy, enthalpy = self._get_total_energy_op(
                    outputs, features, name='energy', verbose=verbose)
                predictions["energy"] = energy
                predictions["enthalpy"] = enthalpy

            if 'forces' in properties or \
                    'stress' in properties or \
                    'total_pressure' in properties:
                with tf.name_scope("Forces"):
                    predictions["forces"] = self._get_forces_op(
                        predictions["energy"],
                        features["positions"],
                        name='forces',
                        verbose=verbose)

            if export_partial_forces:
                with tf.name_scope("PartialForces"):
                    predictions.update(
                        self._get_partial_forces_ops(
                            energy,
                            rij=features["g2.rij"],
                            rijk=features.get("g4.rijk", None)))

            if 'stress' in properties:
                with tf.name_scope("Stress"):
                    voigt_stress, total_stress, total_pressure = \
                        self._get_stress_op(
                            energy=predictions["energy"],
                            cell=features["cell"],
                            volume=features["volume"],
                            positions=features["positions"],
                            forces=predictions["forces"],
                            pulay_stress=features["pulay_stress"],
                            name='stress',
                            verbose=verbose)
                    predictions["stress"] = voigt_stress
                    predictions["total_stress"] = total_stress
                    predictions["total_pressure"] = total_pressure

            if 'hessian' in properties:
                with tf.name_scope("Hessian"):
                    predictions["hessian"] = self._get_hessian_op(
                        predictions["energy"], features["positions"],
                        name='hessian', verbose=verbose)

            if mode == tf_estimator.ModeKeys.PREDICT \
                    and 'elastic' in properties:
                with tf.name_scope("Elastic"):
                    predictions["elastic"] = \
                        elastic_ops.get_elastic_constat_tensor_op(
                            predictions["total_stress"],
                            features["cell"],
                            features["volume"],
                            name='elastic', verbose=verbose)

            return predictions

    def model_fn(self,
                 features: dict,
                 labels: dict,
                 mode: tf_estimator.ModeKeys,
                 params):
        """
        Initialize a model function for `tf_estimator.Estimator`.

        In this method `features` are raw property (positions, cell, etc)
        tensors. Because `tf_estimator.Estimator` requires a `features` as the
        first arg of `model_fn`, we cannot change its name here.

        Parameters
        ----------
        features : dict
            A dict of raw property tensors:
                * 'positions' of shape `[batch_size, n_atoms_max + 1, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'atom_masks' of shape `[batch_size, n_atoms_max + 1]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms_vap' of dtype `int64`.
                * 'etemperature' of dtype `float32` or `float64`.
                * 'pulay_stress' of dtype `float32` or `float64`.
        labels : dict
            A dict of reference tensors.

            Always required:
                * 'energy' of shape `[batch_size, ]`.

            Required if 'forces' should be minimized:
                * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
                  required if 'forces' should be minimized.

            Required if 'stress' or 'total_pressure' should be minimized:
                * 'stress' of shape `[batch_size, 6]` is required if
                  'stress' should be minimized.
                * 'pulay_stress' of shape `[batch_size, ]`

        mode : tf_estimator.ModeKeys
            A `ModeKeys`. Specifies if this is training, evaluation or
            prediction.
        params : EstimatorHyperParams
            Hyperparameters for building and training a `BasicNN`.

        Returns
        -------
        spec : tf_estimator.EstimatorSpec
            Ops and objects returned from a `model_fn` and passed to an
            `Estimator`. `EstimatorSpec` fully defines the model to be run
            by an `Estimator`.

        """
        self._check_keys(features, labels)

        predictions = self.build(features=features,
                                 mode=mode,
                                 verbose=(mode == tf_estimator.ModeKeys.TRAIN))

        if mode == tf_estimator.ModeKeys.PREDICT:
            return tf_estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        total_loss, losses = self.get_total_loss(
            predictions=predictions,
            labels=labels,
            n_atoms=features["n_atoms_vap"],
            atom_masks=features["atom_masks"],
            loss_parameters=params.loss,
            mode=mode)

        ema, train_op = get_train_op(
            losses=losses,
            opt_parameters=params.opt,
            minimize_properties=self._minimize_properties)

        if mode == tf_estimator.ModeKeys.TRAIN:
            training_hooks = get_training_hooks(
                ema=ema,
                train_parameters=params.train,
                num_replicas=params.distribute.num_replicas)
            return tf_estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                              train_op=train_op,
                                              training_hooks=training_hooks)

        eval_metrics_ops = get_eval_metrics_ops(
            eval_properties=self._minimize_properties,
            predictions=predictions,
            labels=labels,
            n_atoms=features["n_atoms_vap"],
            atom_masks=features["atom_masks"],
        )
        evaluation_hooks = get_evaluation_hooks(
            ema=ema,
            train_parameters=params.train)
        return tf_estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          eval_metric_ops=eval_metrics_ops,
                                          evaluation_hooks=evaluation_hooks)

    def export(self, output_graph_path: str, checkpoint=None,
               keep_tmp_files=False, use_ema_variables=True,
               export_partial_forces_model=False):
        """
        Freeze the graph and export the model to a pb file.

        Parameters
        ----------
        output_graph_path : str
            The name of the output graph file.
        checkpoint : str or None
            The tensorflow checkpoint file to restore or None.
        keep_tmp_files : bool
            If False, the intermediate files will be deleted.
        use_ema_variables : bool
            If True, exponentially moving averaged variables will be used.
        export_partial_forces_model : bool
            A boolean. If True, tensoralloy will try to export a model with
            partial forces ops.

        """

        graph = tf.Graph()

        logdir = join(dirname(output_graph_path), 'export')
        if not exists(logdir):
            os.makedirs(logdir)

        input_graph_name = 'input_graph.pb'
        saved_model_ckpt = join(logdir, 'saved_model')
        saved_model_meta = f"{saved_model_ckpt}.meta"

        with graph.as_default():

            if self._transformer is None:
                raise ValueError("A transformer must be attached before "
                                 "exporting to a pb file.")
            elif isinstance(self._transformer, BatchDescriptorTransformer):
                clf = self._transformer.as_descriptor_transformer()
            elif isinstance(self._transformer, DescriptorTransformer):
                serialized = self._transformer.as_dict()
                if 'class' in serialized:
                    serialized.pop('class')
                clf = self._transformer.__class__(**serialized)
            else:
                raise ValueError(f"Unknown transformer: {self._transformer}")

            configs = self.as_dict()
            configs.pop('class')
            nn = self.__class__(**configs)

            if export_partial_forces_model \
                    and isinstance(clf, UniversalTransformer):
                clf.use_computed_dists = False

            nn.attach_transformer(clf)
            predictions = nn.build(clf.get_placeholder_features(),
                                   mode=tf_estimator.ModeKeys.PREDICT,
                                   verbose=True)

            # Encode the JSON dict of the serialized transformer into the graph.
            with tf.name_scope("Transformer/"):
                params_node = tf.constant(
                    json.dumps(clf.as_dict()), name='params')

            # Add a timestamp to the graph
            with tf.name_scope("Metadata/"):
                timestamp_node = tf.constant(
                    str(datetime.today()), name='timestamp')
                y_atomic_node = tf.constant(nn._get_atomic_energy_tensor_name(),
                                            name="y_atomic")
                fp_prec_node = tf.constant(
                    get_float_precision().name, name='precision')
                tf_version_node = tf.constant(tf.__version__, name='tf_version')

                ops = {key: tensor.name for key, tensor in predictions.items()}
                ops_node = tf.constant(json.dumps(ops), name='ops')

            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                if use_ema_variables:
                    # Restore the moving averaged variables
                    ema = tf.train.ExponentialMovingAverage(
                        Defaults.variable_moving_average_decay)
                    saver = tf.train.Saver(ema.variables_to_restore())
                else:
                    saver = tf.train.Saver(var_list=tf.trainable_variables())
                if checkpoint is not None:
                    saver.restore(sess, checkpoint)

                # Create another saver to save the trainable variables
                saver = tf.train.Saver(var_list=tf.model_variables())
                checkpoint_path = saver.save(
                    sess, saved_model_ckpt, global_step=0)
                graph_io.write_graph(graph_or_graph_def=graph,
                                     logdir=logdir,
                                     name=input_graph_name)

            input_graph_path = join(logdir, input_graph_name)
            input_saver_def_path = ""
            input_binary = False
            restore_op_name = "save/restore_all"
            filename_tensor_name = "save/Const:0"
            clear_devices = True
            input_meta_graph = saved_model_meta

            output_node_names = [
                params_node.op.name,
                timestamp_node.op.name,
                y_atomic_node.op.name,
                fp_prec_node.op.name,
                tf_version_node.op.name,
                ops_node.op.name,
            ]

            for tensor in predictions.values():
                if not is_tensor(tensor):
                    continue
                output_node_names.append(tensor.op.name)

            for node in graph.as_graph_def().node:
                name = node.name
                if name.startswith('Placeholders/'):
                    output_node_names.append(name)

            freeze_graph.freeze_graph(
                input_graph_path, input_saver_def_path, input_binary,
                checkpoint_path, ",".join(output_node_names), restore_op_name,
                filename_tensor_name, output_graph_path, clear_devices, "", "",
                input_meta_graph)

        if not keep_tmp_files:
            shutil.rmtree(logdir, ignore_errors=True)

        tf.logging.info(f"Model exported to {output_graph_path}")
