# coding=utf-8
"""
This module defines the basic neural network for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
import shutil

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union
from os.path import join, dirname
from tensorflow_core.python.tools import freeze_graph
from tensorflow_core.python.framework import graph_io
from tensorflow_core.python.framework.tensor_util import is_tensor
from tensorflow_estimator.python.estimator.estimator_lib import EstimatorSpec
from ase.units import GPa

from tensoralloy.utils import GraphKeys, Defaults, safe_select, ModeKeys
from tensoralloy.nn.dataclasses import StructuralProperty, LossParameters
from tensoralloy.nn.dataclasses import EnergyOps
from tensoralloy.nn.utils import log_tensor, is_first_replica
from tensoralloy.nn.opt import get_train_op, get_training_hooks
from tensoralloy.nn.eval import get_evaluation_hooks
from tensoralloy.nn import losses as loss_ops
from tensoralloy.nn.constraint import elastic as elastic_ops
from tensoralloy.nn.constraint import rose as rose_ops
from tensoralloy.nn.constraint import eentropy as eentropy_ops
from tensoralloy.nn.constraint import fc as fc_ops
from tensoralloy.nn.constraint import extra_db as extra_db_ops
from tensoralloy.transformer.base import BaseTransformer
from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.transformer.universal import BatchUniversalTransformer
from tensoralloy.precision import get_float_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


API_VERSION = "1.1"


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
    StructuralProperty(name='eentropy'),
    StructuralProperty(name='free_energy'),
    StructuralProperty(name='atomic', exportable=True, minimizable=False),
    StructuralProperty(name='forces'),
    StructuralProperty(name='stress'),
    StructuralProperty(name='total_pressure'),
    StructuralProperty(name='hessian', minimizable=False),
    StructuralProperty(name='elastic'),
    StructuralProperty(name='rose', exportable=False),
    StructuralProperty(name='eentropy/c', exportable=False),
    StructuralProperty(name='hessian/c', exportable=False),
    StructuralProperty(name='extra/c', exportable=False)
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

    # Global scope for this potential model.
    scope = "Basic"

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

    @property
    def variational_energy(self) -> str:
        """
        The energy op corresponding to forces. By default E(sigma->0)
        corresponds to forces.

        For finite temperature models with Fermi-Dirac smearing, free_energy
        should be used as variational energy.
        """
        if self.is_finite_temperature:
            return "free_energy"
        else:
            return "energy"

    @property
    def is_finite_temperature(self) -> bool:
        """
        Return True if this is a finite temperature model.
        """
        return False

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
    def transformer(self) -> UniversalTransformer:
        """
        Get the attached transformer.
        """
        return self._transformer

    def attach_transformer(self, clf: UniversalTransformer):
        """
        Attach a descriptor transformer to this potential.
        """
        self._transformer = clf

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        raise NotImplementedError("This method must be overridden!")

    def _get_energy_ops(self,
                        outputs,
                        features: dict,
                        verbose=True) -> EnergyOps:
        """
        Return the Ops to compute different types of energies:
            * energy: the internal energy U
            * entropy: the electron entropy (unitless) S
            * free_energy: the electron free energy E(F) = U - T*S

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
        name : str
            The name of the output potential energy tensor.
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        raise NotImplementedError("This method must be overridden!")

    @staticmethod
    def _get_forces_op(energy, positions, verbose=True):
        """
        Return the Op to compute atomic forces (eV / Angstrom).
        """
        dEdR = tf.gradients(energy, positions, name='dEdR')[0]
        # Setup the splitting axis. `energy` may be a 1D vector or a scalar.
        axis = energy.shape.ndims
        # Please remember: f = -dE/dR
        forces = tf.negative(
            tf.split(dEdR, [1, -1], axis=axis, name='split')[1],
            name='forces')
        if verbose:
            log_tensor(forces)
        return forces

    @staticmethod
    def _get_full_stress_and_virial(energy: tf.Tensor, cell, volume,
                                    positions, forces):
        """
        Return the Ops to compute the 3x3 virial and stress tensors.

            virial = stress * volume = -F^T @ R + (dE/dh)^T @ h

        where `E` denotes the total energy, `h` is the 3x3 row-major lattice
        matrix, `R` and `F` are positions and total forces of the atoms.
        """
        with tf.name_scope("Full"):
            # The cell tensors in Python/ASE are row-major. So `dE/dh` must be
            # transposed.
            dEdh = tf.identity(tf.gradients(energy, cell)[0], name='dEdh')
            if cell.shape.ndims == 2:
                with tf.name_scope("Right"):
                    right = tf.matmul(tf.transpose(dEdh, name='dEdhT'), cell,
                                      name='right')
                with tf.name_scope("Left"):
                    positions = tf.split(
                        positions, [1, -1], axis=0, name='split')[1]
                    left = tf.matmul(tf.transpose(forces), positions)
                    left = tf.negative(left, name='left')
                virial = tf.add(left, right, name='virial')
                stress = tf.math.truediv(virial, volume, name='ase')
            else:
                with tf.name_scope("Right"):
                    right = tf.einsum('ikj,ikl->ijl', dEdh, cell, name='right')
                with tf.name_scope("Left"):
                    positions = tf.split(
                        positions, [1, -1], axis=1, name='split')[1]
                    left = tf.einsum('ijk,ijl->ijlk', positions, forces)
                    left = tf.reduce_sum(left, axis=1, keepdims=False)
                    left = tf.negative(left, name='left')
                virial = tf.add(left, right, name='virial')
                stress = tf.math.truediv(tf.reshape(virial, (-1, 9)),
                                         tf.reshape(volume, (-1, 1)))
                stress = tf.reshape(stress, (-1, 3, 3), name='ase')
            return stress, virial

    @staticmethod
    def _convert_to_voigt_stress(stress, batch_size, verbose=False):
        """
        Convert a 3x3 stress tensor or a Nx3x3 stress tensor to corresponding
        Voigt form.
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
            stress = tf.gather_nd(stress, voigt, name='stress')
            if verbose:
                log_tensor(stress)
            return stress

    def _get_stress_ops(self, energy: tf.Tensor, cell, volume, positions,
                        forces, return_pressure=False, verbose=True):
        """
        Return stress Ops:

            voigt_stress: the stress (eV/Ang^3) in Voigt form.
            virial: the 3x3 virial tensor (eV)
            total_pressure: the total pressure scalar (GPa)

        """

        # Get the 3x3 full stress tensor
        stress, virial = self._get_full_stress_and_virial(
            energy=energy,
            cell=cell,
            volume=volume,
            positions=positions,
            forces=forces)
        if verbose:
            log_tensor(virial)

        # Get the Voigt stress tensor
        ndims = virial.shape.ndims
        batch_size = cell.shape[0].value or energy.shape[0].value
        if ndims == 3 and batch_size is None:
            raise ValueError("The batch size cannot be inferred.")
        voigt_stress = self._convert_to_voigt_stress(
            stress, batch_size, verbose=verbose)

        if return_pressure:
            total_pressure = self._get_total_pressure_op(
                stress, verbose=verbose)
        else:
            total_pressure = tf.no_op()

        return voigt_stress, virial, total_pressure

    @staticmethod
    def _get_total_pressure_op(stress: tf.Tensor, name='pressure',
                               verbose=True):
        """
        Return the Op to compute the reduced total pressure (GPa).

            total_pressure = -trace(stress) / 3.0

        """
        dtype = stress.dtype
        total_pressure = tf.math.truediv(tf.trace(stress),
                                         tf.constant(-3.0 * GPa, dtype=dtype),
                                         name=f"{name}/GPa")
        if verbose:
            log_tensor(total_pressure)
        return total_pressure

    @staticmethod
    def _get_hessian_op(energy: tf.Tensor, positions: tf.Tensor, verbose=True):
        """
        Return the Op to compute the Hessian matrix:

            hessian = d^2E / dR^2 = d(dE / dR) / dR

        """
        hessian = tf.identity(tf.hessians(energy, positions)[0], name='hessian')
        if verbose:
            log_tensor(hessian)
        return hessian

    def _get_energy_loss(self,
                         predictions,
                         labels,
                         n_atoms,
                         max_train_steps,
                         loss_parameters: LossParameters,
                         collections,
                         sample_weight=None,
                         normalized_weight=False) -> Dict[str, tf.Tensor]:
        """
        Return energy loss(es).
        """
        loss = loss_ops.get_energy_loss(
            labels=labels["energy"],
            predictions=predictions["energy"],
            n_atoms=n_atoms,
            max_train_steps=max_train_steps,
            options=loss_parameters.energy,
            sample_weight=sample_weight,
            normalized_weight=normalized_weight,
            collections=collections)
        return {"energy": loss}

    def get_total_loss(self,
                       predictions,
                       labels,
                       n_atoms,
                       atom_masks,
                       loss_parameters: LossParameters,
                       max_train_steps=None,
                       mode=ModeKeys.TRAIN):
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
                * 'virial' of shape `[batch_size, 3, 3]` with unit `eV`.

            For finite temperature systems those are required:
                * 'eentropy' (electron entropy) of shape `[batch_size, ]`
                * 'free_energy' (electron free energy) of shape `[batch_size, ]`

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

        n_atoms : tf.Tensor
            A `int64` tensor of shape `[batch_size, ]`.
        atom_masks : tf.Tensor
            A float tensor of shape `[batch_size, n_atoms_max + 1]`.
        max_train_steps : int
            The maximum number of training steps.
        loss_parameters : LossParameters
            The hyper-parameters for computing the total loss.
        mode : ModeKeys
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
            
            if 'forces' in self._minimize_properties and \
                    loss_parameters.adaptive_sample_weight.enabled:
                options = loss_parameters.adaptive_sample_weight
                sample_weight = loss_ops.adaptive_sample_weight(
                    labels["forces"], 
                    n_atoms, 
                    options.metric,
                    options.method, 
                    *options.params)
                normalized_weight = options.normalized
            else:
                sample_weight = None
                normalized_weight = False

            losses = self._get_energy_loss(
                predictions=predictions,
                labels=labels,
                n_atoms=n_atoms,
                max_train_steps=max_train_steps,
                loss_parameters=loss_parameters,
                collections=collections,
                sample_weight=sample_weight, 
                normalized_weight=normalized_weight)

            if 'forces' in self._minimize_properties:
                losses["forces"] = loss_ops.get_forces_loss(
                    labels=labels["forces"],
                    predictions=predictions["forces"],
                    atom_masks=atom_masks,
                    max_train_steps=max_train_steps,
                    options=loss_parameters.forces,
                    collections=collections,
                    sample_weight=sample_weight,
                    normalized_weight=normalized_weight)

            if 'total_pressure' in self._minimize_properties:
                losses["total_pressure"] = loss_ops.get_pressure_loss(
                    labels=labels["total_pressure"],
                    predictions=predictions["total_pressure"],
                    max_train_steps=max_train_steps,
                    options=loss_parameters.total_pressure,
                    collections=collections)

            if 'stress' in self._minimize_properties:
                losses["stress"] = loss_ops.get_stress_loss(
                    labels=labels["stress"],
                    predictions=predictions["stress"],
                    max_train_steps=max_train_steps,
                    options=loss_parameters.stress,
                    collections=collections,
                    sample_weight=sample_weight,
                    normalized_weight=normalized_weight)

            l2_loss = loss_ops.get_l2_regularization_loss(
                options=loss_parameters.l2,
                collections=collections)
            if l2_loss is not None:
                losses['l2'] = l2_loss

            verbose = bool(mode == ModeKeys.TRAIN)

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

            if 'eentropy/c' in self._minimize_properties:
                if loss_parameters.eentropy_constraint.crystals is not None:
                    loss = eentropy_ops.get_eentropy_constraint_loss(
                        base_nn=self,
                        options=loss_parameters.eentropy_constraint,
                        verbose=verbose)
                    if loss is not None:
                        losses['eentropy/c'] = loss

            if 'hessian/c' in self._minimize_properties:
                if loss_parameters.hessian_constraint.crystals is not None:
                    loss = fc_ops.get_fc2_loss(
                        self,
                        options=loss_parameters.hessian_constraint,
                        verbose=verbose)
                    if loss is not None:
                        losses['hessian/c'] = loss
            
            if 'extra/c' in self._minimize_properties:
                if Path(loss_parameters.extra_constraint.filename).exists():
                    loss = extra_db_ops.get_extra_db_constraint_loss(
                        self,
                        options=loss_parameters.extra_constraint,
                        max_train_steps=max_train_steps,
                        verbose=verbose)
                    if loss is not None:
                        losses['extra/c'] = loss

            for tensor in losses.values():
                tf.summary.scalar(tensor.op.name + '/summary', tensor)

        total_loss = tf.add_n(list(losses.values()), name='total_loss')

        if is_first_replica():
            tf.add_to_collection(GraphKeys.TRAIN_METRICS, total_loss)

        return total_loss, losses

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
                           mode: ModeKeys,
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
        descriptors : dict
            A dict of Ops to get atomic descriptors. This should be produced by
            an overrided `BaseTransformer.get_descriptors()`.
        mode : ModeKeys
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
            if prop in ('elastic', 'rose', 'eentropy/c', 'hessian/c', 
                        'extra/c'):
                continue
            assert prop in labels, f"{prop} is missing"

    def build(self,
              features: dict,
              mode=ModeKeys.TRAIN,
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
        mode : ModeKeys
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
                * 'virial' of shape `[batch_size, 3, 3]` with unit `eV`.

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

        if ModeKeys.for_prediction(mode):
            properties = self._export_properties
            if self.is_finite_temperature:
                properties.extend(["eentropy", "free_energy"])
        else:
            properties = self._minimize_properties

        with tf.name_scope("Output"):

            with tf.name_scope("Energy"):
                ops = self._get_energy_ops(outputs, features, verbose)
                predictions = ops.as_dict()

            if 'forces' in properties or \
                    'stress' in properties or \
                    'total_pressure' in properties:
                with tf.name_scope("Forces"):
                    predictions["forces"] = self._get_forces_op(
                        energy=predictions[self.variational_energy],
                        positions=features["positions"],
                        verbose=verbose)

            if 'stress' in properties:
                with tf.name_scope("Stress"):
                    voigt_stress, virial, total_pressure = \
                        self._get_stress_ops(
                            energy=predictions[self.variational_energy],
                            cell=features["cell"],
                            volume=features["volume"],
                            positions=features["positions"],
                            forces=predictions["forces"],
                            verbose=verbose)
                    predictions["stress"] = voigt_stress
                    predictions["virial"] = virial
                    predictions["total_pressure"] = total_pressure

            if 'hessian' in properties:
                with tf.name_scope("Hessian"):
                    predictions["hessian"] = self._get_hessian_op(
                        energy=predictions[self.variational_energy],
                        positions=features["positions"],
                        verbose=verbose)

            if mode == ModeKeys.PREDICT and 'elastic' in properties:
                with tf.name_scope("Elastic"):
                    predictions["elastic"] = \
                        elastic_ops.get_elastic_constat_tensor_op(
                            predictions["virial"],
                            features["cell"],
                            features["volume"],
                            name='elastic', verbose=verbose)

            return predictions

    def get_eval_metrics(self, labels, predictions, n_atoms, atom_masks):
        """
        Return a dict of Ops as the evaluation metrics.

        Always required:
            * 'energy' of shape `[batch_size, ]`

        Required if finite temperature:
            * 'eentropy' of shape `[batch_size, ]`
            * 'free_energy' of shape `[batch_size, ]`

        Required if 'forces' should be minimized:
            * 'forces' of shape `[batch_size, n_atoms_max + 1, 3]` is
              required if 'forces' should be minimized.
            * 'atom_masks' of shape `[batch_size, n_atoms_max + 1]`

        Required if 'stress' or 'total_pressure' should be minimized:
            * 'stress' of shape `[batch_size, 6]` is required if
              'stress' should be minimized.
            * 'total_pressure' of shape `[batch_size, ]`

        `n_atoms` is a `int64` tensor with shape `[batch_size, ]`, representing
        the number of atoms in each structure.

        """
        with tf.name_scope("Metrics"):
            n_atoms = tf.cast(n_atoms, labels["energy"].dtype, name='n_atoms')

            metrics = self._get_eval_energy_metrics(
                labels, predictions, n_atoms)
            if 'forces' in self._minimize_properties:
                metrics.update(self._get_eval_forces_metrocs(
                    labels, predictions, atom_masks))
            if 'stress' in self._minimize_properties:
                metrics.update(self._get_eval_stress_metrics(
                    labels, predictions))
            metrics.update(self._get_eval_constraints_metrics())

            return metrics

    def _get_eval_energy_metrics(self, labels, predictions, n_atoms):
        """
        Default evaluation metrics for energy predictions.
        """
        with tf.name_scope("Energy"):
            name_map = {
                'energy': 'U',
                'free_energy': 'F',
                'eentropy': 'S',
            }
            metrics = {}
            for prop, desc in name_map.items():
                if prop in self._minimize_properties:
                    x = labels[prop]
                    y = predictions[prop]
                    xn = x / n_atoms
                    yn = y / n_atoms
                    ops_dict = {
                        f'{desc}/mae': tf.metrics.mean_absolute_error(x, y),
                        f'{desc}/mse': tf.metrics.mean_squared_error(x, y),
                        f'{desc}/mae/atom': tf.metrics.mean_absolute_error(
                            xn, yn)}
                    metrics.update(ops_dict)
            return metrics

    @staticmethod
    def _get_eval_forces_metrocs(labels, predictions, atom_masks):
        """
        Default evaluation metrics for forces predictions.
        """
        with tf.name_scope("Forces"):
            with tf.name_scope("Split"):
                x = tf.split(labels["forces"], [1, -1], axis=1)[1]
                mask = tf.cast(tf.split(atom_masks, [1, -1], axis=1)[1],
                               tf.bool)
            x = tf.boolean_mask(x, mask, axis=0, name='x')
            y = tf.boolean_mask(
                predictions["forces"], mask, axis=0, name='y')
            with tf.name_scope("Flatten"):
                x = tf.reshape(x, (-1, ), name='x')
                y = tf.reshape(y, (-1, ), name='y')
            return {'Forces/mae': tf.metrics.mean_absolute_error(x, y),
                    'Forces/mse': tf.metrics.mean_squared_error(x, y)}

    @staticmethod
    def _get_eval_stress_metrics(labels, predictions):
        """
        Default evaluation metrics for stress predictions.
        """
        with tf.name_scope("Stress"):
            x = labels["stress"]
            y = predictions["stress"]
            ops_dict = {
                'Stress/mae': tf.metrics.mean_absolute_error(x, y),
                'Stress/mse': tf.metrics.mean_squared_error(x, y)}
            with tf.name_scope("rRMSE"):
                upper = tf.linalg.norm(x - y, axis=-1)
                lower = tf.linalg.norm(x, axis=-1)
                ops_dict['Stress/Rel'] = \
                    tf.metrics.mean(upper / lower)
            return ops_dict

    @staticmethod
    def _get_eval_constraints_metrics():
        """
        Default evaluation metrics for constraints.
        """
        metrics = {}
        for tensor in tf.get_collection(GraphKeys.EVAL_METRICS):
            slist = tensor.op.name.split("/")
            if "Elastic" in tensor.op.name:
                istart = slist.index('Elastic')
                key = "/".join(slist[istart:])
                metrics[key] = (tensor, tf.no_op())
            elif "Rose" in tensor.op.name:
                istart = slist.index('Rose')
                istop = slist.index('EOS')
                key = "/".join(slist[istart: istop])
                metrics[key] = (tensor, tf.no_op())
            elif "Diff" in tensor.op.name:
                if 'pred' in tensor.op.name:
                    key = "/".join(slist[-3:])
                else:
                    istart = slist.index('Diff')
                    istop = slist.index('mae')
                    key = "/".join(slist[istart: istop - 1])
                metrics[key] = (tensor, tf.no_op())
            elif "Extra/total_loss" in tensor.op.name:
                metrics["extra"] = (tensor, tf.no_op())
        return metrics

    def model_fn(self,
                 features: dict,
                 labels: dict,
                 mode: ModeKeys,
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

        mode : ModeKeys
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
                                 verbose=(mode == ModeKeys.TRAIN))

        if mode == ModeKeys.PREDICT:
            return EstimatorSpec(mode=mode, predictions=predictions)

        total_loss, losses = self.get_total_loss(
            predictions=predictions,
            labels=labels,
            n_atoms=features["n_atoms_vap"],
            atom_masks=features["atom_masks"],
            max_train_steps=params.train.train_steps,
            loss_parameters=params.loss,
            mode=mode)

        ema, train_op = get_train_op(
            losses=losses,
            opt_parameters=params.opt,
            minimize_properties=self._minimize_properties)

        if mode == ModeKeys.TRAIN:
            training_hooks = get_training_hooks(
                ema=ema,
                train_parameters=params.train,
                num_replicas=params.distribute.num_replicas)
            return EstimatorSpec(mode=mode,
                                 loss=total_loss,
                                 train_op=train_op,
                                 training_hooks=training_hooks)

        eval_metrics_ops = self.get_eval_metrics(
            predictions=predictions,
            labels=labels,
            n_atoms=features["n_atoms_vap"],
            atom_masks=features["atom_masks"],
        )
        evaluation_hooks = get_evaluation_hooks(
            ema=ema,
            train_parameters=params.train)
        return EstimatorSpec(mode=mode,
                             loss=total_loss,
                             eval_metric_ops=eval_metrics_ops,
                             evaluation_hooks=evaluation_hooks)

    def export(self, output_graph_path: str, checkpoint=None,
               keep_tmp_files=False, use_ema_variables=True,
               mode=ModeKeys.PREDICT, **kwargs):
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
        mode : ModeKeys
            The export mode: `infer`, `lammps` or `kmc`

        """
        if mode == ModeKeys.NATIVE:
            self.export_to_lammps_native()
            return

        graph = tf.Graph()

        logdir = Path(dirname(output_graph_path)).joinpath('export')
        if not logdir.exists():
            logdir.mkdir()

        input_graph_name = 'input_graph.pb'
        saved_model_ckpt = logdir.joinpath('saved_model')
        saved_model_meta = f"{saved_model_ckpt}.meta"

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
            predictions = nn.build(clf.get_placeholder_features(),
                                   mode=mode,
                                   verbose=True)

            # Encode the JSON dict of the serialized transformer into the graph.
            with tf.name_scope("Transformer/"):
                params_node = tf.constant(
                    json.dumps(clf.as_dict()), name='params')

            # Add a timestamp to the graph
            with tf.name_scope("Metadata/"):
                timestamp_node = tf.constant(
                    str(datetime.today()), name='timestamp')
                fp_prec_node = tf.constant(
                    get_float_precision().name, name='precision')
                tf_version_node = tf.constant(tf.__version__, name='tf_version')
                variational_energy_node = tf.constant(
                    nn.variational_energy, name='variational_energy')
                is_finite_temperature_node = tf.constant(
                    int(nn.is_finite_temperature), name="is_finite_temperature")
                api_version_node = tf.constant(API_VERSION, name="api")
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
                    saver = tf.train.Saver(var_list=tf.model_variables())
                if checkpoint is not None:
                    saver.restore(sess, checkpoint)

                # Create another saver to save the trainable variables
                saver = tf.train.Saver(var_list=tf.global_variables())
                checkpoint_path = saver.save(
                    sess, saved_model_ckpt, global_step=0)
                graph_io.write_graph(graph_or_graph_def=graph,
                                     logdir=str(logdir),
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
                fp_prec_node.op.name,
                tf_version_node.op.name,
                is_finite_temperature_node.op.name,
                variational_energy_node.op.name,
                api_version_node.op.name,
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
