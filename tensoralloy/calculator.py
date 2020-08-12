#!coding=utf-8
"""
This module defines various neural network based calculators.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json

from os.path import dirname
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import GPa
from ase.calculators.calculator import all_changes
from tensorflow.python import debug as tf_debug
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from typing import List, Tuple

from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer import UniversalTransformer
from tensoralloy.nn.basic import exportable_properties
from tensoralloy.precision import precision_scope

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TensorAlloyCalculator(Calculator):
    """
    ASE-Calculator for TensorAlloy derived protobuf models.
    """

    implemented_properties = [prop.name for prop in exportable_properties]
    default_parameters = {}
    nolabel = True

    def __init__(self, graph_model_path: str, atoms=None, serial_mode=False):
        """
        Initialization method.

        Parameters
        ----------
        graph_model_path : str
            The exported model to load.
        atoms : Atoms
            The target `Atoms` object.
        serial_mode : bool
            If True, the program will only use 1 core and 1 thread.

        """
        super(TensorAlloyCalculator, self).__init__(
            restart=None, ignore_bad_restart_file=False, label=None,
            atoms=atoms)

        graph = tf.Graph()

        with graph.as_default():

            output_graph_def = graph_pb2.GraphDef()
            with open(graph_model_path, "rb") as fp:
                output_graph_def.ParseFromString(fp.read())
                importer.import_graph_def(output_graph_def, name="")

            self._graph_model_path = graph_model_path
            self._model_dir = dirname(graph_model_path)

            if serial_mode:
                config = tf.ConfigProto(device_count={'CPU': 1})
                config.inter_op_parallelism_threads = 1
                config.intra_op_parallelism_threads = 1
            else:
                config = tf.ConfigProto()

            config.allow_soft_placement = True

            self._sess = tf.Session(config=config, graph=graph)
            self._graph = graph
            self._transformer = self._get_transformer()
            self._ops, self._fp_precision = self._get_ops()
            self.implemented_properties = self._predict_properties
            self._ncalls = 0

    @property
    def elements(self) -> List[str]:
        """
        Return a list of str as the supported elements.
        """
        return self._transformer.elements

    @property
    def transformer(self) -> DescriptorTransformer:
        """
        Return the `DescriptorTransformer` of this calculator.
        """
        return self._transformer

    @property
    def predict_properties(self):
        """
        Return a list of str as the predictable properties.
        """
        return self._predict_properties

    def get_model_timestamp(self):
        """
        Return the timestamp when the graph model was generated or None.
        """
        try:
            op = self._graph.get_tensor_by_name('Metadata/timestamp:0')
        except Exception:
            return None
        return self._sess.run(op).decode('utf-8')

    def _get_transformer(self):
        """
        Recover a `DescriptorTransformer` from the graph.
        """
        params = json.loads(self._sess.run(
            self._graph.get_tensor_by_name('Transformer/params:0')))
        if 'predict_properties' in params:
            self._predict_properties = params.pop('predict_properties')
        else:
            self._predict_properties = []
        cls = params.pop('class')
        if cls == 'UniversalTransformer':
            return UniversalTransformer(**params)
        else:
            raise ValueError(f"Unsupported transformer: {cls}")

    def _get_ops(self):
        """
        Return a dict of output Ops.
        """
        graph = self._graph
        ops = json.loads(
            self._sess.run(self._graph.get_tensor_by_name("Metadata/ops:0")))
        ops = {prop: graph.get_tensor_by_name(name)
               for prop, name in ops.items() if name.endswith(":0")}
        self._predict_properties = list(ops.keys())
        for name, tensor in ops.items():
            if tensor.dtype == tf.float32:
                fp_precision = 'medium'
            else:
                fp_precision = 'high'
            break
        else:
            raise Exception("Validated Ops cannot be found")
        return ops, fp_precision

    def get_magnetic_moment(self, atoms=None):
        """
        This calculator cannot predict magnetic moments.
        """
        return None

    def get_magnetic_moments(self, atoms=None):
        """
        This calculator cannot predict magnetic moments.
        """
        return None

    def get_enthalpy(self, atoms=None):
        """
        Return the enthalpy energy (eV).
        """
        return self.get_property('enthalpy', atoms=atoms)

    def get_pv_energy(self, atoms=None):
        """
        Return the PV energy (eV).
        """
        return self.get_property('pv', atoms=atoms)

    def get_electron_entropy(self, atoms=None):
        """
        Return the electron entropy S.
        """
        return self.get_property('eentropy', atoms=atoms)

    def get_free_energy(self, atoms=None):
        """
        Return the free energy for finite temperature system.
        F = U - T*S
        """
        return self.get_property('free_energy', atoms=atoms)

    def get_atomic_energy(self, atoms=None):
        """
        Return an array as the atomic energies.
        """
        values = self.get_property('atomic', atoms=atoms)
        values = np.insert(values, 0, 0, 0)
        clf = self.transformer.get_vap_transformer(atoms)
        return clf.map_array(values.reshape((-1, 1)), reverse=True).flatten()

    def get_hessian(self, atoms=None):
        """
        Return the Hessian matrix.

        Returns
        -------
        hessian : array_like
            The second-order derivatives of E w.r.t R, the Hessian matrix.
            The shape is [3 * N, 3 * N].

        """
        hessian = self.get_property('hessian', atoms)
        clf = self.transformer.get_vap_transformer(atoms)
        return clf.reverse_map_hessian(hessian)

    def get_forces(self, atoms=None):
        """
        Return the atomic forces.
        """
        forces = np.insert(self.get_property('forces', atoms), 0, 0, 0)
        clf = self.transformer.get_vap_transformer(atoms)
        return clf.map_forces(forces, reverse=True)

    def get_stress(self, atoms=None, voigt=True):
        """
        Return the stress tensor.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms` object.
        voigt : bool
            If True, return the stress tensor in Voigt order. Otherwise the 3x3
            matrix will be returned.

        Returns
        -------
        stress : array_like
            The stress tensor in ASE internal unit, 'eV/Angstrom**3'.

        """
        if atoms is None:
            atoms = self.atoms
        stress = self.get_property('stress', atoms)
        if not voigt:
            xx, yy, zz, yz, xz, xy = stress
            stress = np.array([[xx, xy, xz],
                               [xy, yy, yz],
                               [xz, yz, zz]])
        return stress
    
    def get_total_pressure(self, atoms=None):
        """
        Return the external pressure of the target `Atoms`.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms`.

        Returns
        -------
        total_pressure : float
            The total pressure, in GPa.

        """
        stress = self.get_stress(atoms)
        return np.mean(stress[:3]) * (-1.0) / GPa

    def get_elastic_constant_tensor(self, atoms=None):
        """
        Return the elastic constant tensor C.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms`.

        Returns
        -------
        elastic_tensor : array_like
            The elastic constant tensor.

        References
        ----------
        https://wiki.materialsproject.org/Elasticity_calculations

        """
        atoms = atoms or self.atoms
        assert atoms.pbc.all()
        elastic = self.get_property('elastic', atoms, allow_calculation=True)
        for i in range(6):
            for j in range(i + 1, 6):
                elastic[j, i] = elastic[i, j]
        return elastic

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes, debug_mode=False):
        """
        Calculate the total energy and other properties (1body, kbody, atomic).

        Parameters
        ----------
        atoms : `Atoms`
            An `ase.Atoms` object to calculate.
        properties : Tuple[str]
            A list of str as the properties to calculate. Available options
            are: 'energy', 'atomic', '1body' and 'kbody'.
        system_changes: List[str]
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.
        debug_mode : bool
            Use the debug mode if True.

        """
        Calculator.calculate(self, atoms, properties, system_changes)
        with precision_scope(self._fp_precision):
            with self._graph.as_default():
                ops = {target: self._ops[target] for target in properties}
                if debug_mode:
                    sess = tf_debug.LocalCLIDebugWrapperSession(
                        self._sess, ui_type="readline")
                else:
                    sess = self._sess
                self.results = sess.run(
                    ops, feed_dict=self._transformer.get_feed_dict(atoms))
                self._ncalls += 1

    def reset_call_counter(self):
        """
        Reset the `ncall` counter.
        """
        self._ncalls = 0

    @property
    def ncalls(self):
        """
        Return the accummulative number of `calculate` calls.
        """
        return self._ncalls
