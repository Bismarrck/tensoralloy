# coding=utf-8
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
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from typing import List, Tuple

from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer import SymmetryFunctionTransformer
from tensoralloy.nn.basic import all_properties
from tensoralloy.precision import precision_scope

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TensorAlloyCalculator(Calculator):
    """
    ASE-Calculator for TensorAlloy derived protobuf models.
    """

    implemented_properties = [prop.name for prop in all_properties]
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
        if 'class' in params:
            params.pop('class')
        if 'predict_properties' in params:
            self._predict_properties = params.pop('predict_properties')
        else:
            self._predict_properties = []
        return SymmetryFunctionTransformer(**params)

    def _get_ops(self):
        """
        Return a dict of output Ops.
        """
        graph = self._graph
        props_and_names = {
            'energy': 'Output/Energy/energy:0',
            'forces': 'Output/Forces/forces:0',
            'stress': 'Output/Stress/Voigt/stress:0',
        }
        ops = {}
        for prop, name in props_and_names.items():
            try:
                tensor = graph.get_tensor_by_name(name)
            except KeyError:
                continue
            else:
                ops[prop] = tensor
        self._predict_properties = list(ops.keys())
        if ops['energy'].dtype == tf.float32:
            fp_precision = 'medium'
        else:
            fp_precision = 'high'
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

    def calculate(self, atoms=None, properties=('energy', 'forces'), *args):
        """
        Calculate the total energy and other properties (1body, kbody, atomic).

        Parameters
        ----------
        atoms : `Atoms`
            An `ase.Atoms` object to calculate.
        properties : Tuple[str]
            A list of str as the properties to calculate. Available options
            are: 'energy', 'atomic', '1body' and 'kbody'.

        """
        Calculator.calculate(self, atoms, properties, *args)
        with precision_scope(self._fp_precision):
            with self._graph.as_default():
                ops = {target: self._ops[target] for target in properties}
                self.results = self._sess.run(
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
