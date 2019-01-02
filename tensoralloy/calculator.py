# coding=utf-8
"""
This module defines various neural network based calculators.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
from ase import Atoms
from ase.calculators.calculator import Calculator
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from typing import List, Tuple

from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer import SymmetryFunctionTransformer
from tensoralloy.nn.basic import exportable_properties

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class TensorAlloyCalculator(Calculator):
    """
    ASE-Calculator for TensorAlloy derived protobuf models.
    """

    implemented_properties = [prop.name for prop in exportable_properties]
    default_parameters = {}
    nolabel = True

    def __init__(self, graph_path: str, label=None, atoms=None):
        """
        Initialization method.
        """
        super(TensorAlloyCalculator, self).__init__(
            restart=None, ignore_bad_restart_file=False, label=label,
            atoms=atoms)

        graph = tf.Graph()

        with graph.as_default():

            output_graph_def = graph_pb2.GraphDef()
            with open(graph_path, "rb") as fp:
                output_graph_def.ParseFromString(fp.read())
                importer.import_graph_def(output_graph_def, name="")

            self._sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True),
                graph=graph)
            self._graph = graph
            self._transformer = self._get_transformer()
            self._ops = self._get_ops()

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

    def _get_transformer(self):
        """
        Recover a `DescriptorTransformer` from the graph.
        """
        params = json.loads(self._sess.run(
            self._graph.get_tensor_by_name('Transformer/params:0')))
        if 'class' in params:
            cls = params.pop('class')
        else:
            cls = 'SymmetryFunctionTransformer'
        if 'predict_properties' in params:
            self._predict_properties = params.pop('predict_properties')
        else:
            self._predict_properties = []
        if cls == 'SymmetryFunctionTransformer':
            return SymmetryFunctionTransformer(**params)
        else:
            raise ValueError(f"Unknown transformer: {cls}")

    def _get_ops(self):
        """
        Return a dict of output Ops.
        """
        graph = self._graph
        props_and_names = {
            'energy': 'Output/Energy/energy:0',
            'forces': 'Output/Forces/forces:0',
            'stress': 'Output/Stress/Voigt/stress:0',
            'total_pressure': 'Output/Pressure/pressure:0',
            'hessian': 'Output/Hessian/hessian:0',
        }
        ops = {}
        for prop, name in props_and_names.items():
            try:
                ops[prop] = graph.get_tensor_by_name(name)
            except KeyError:
                continue
        if not self._predict_properties:
            self._predict_properties = list(ops.keys())
        else:
            assert set(self._predict_properties) == set(ops.keys())
        return ops

    def get_hessian(self, atoms=None):
        """
        Return the Hessian matrix.
        """
        hessian = self.get_property('hessian', atoms)
        clf = self.transformer.get_index_transformer(atoms)
        return clf.reverse_map_hessian(hessian)

    def get_forces(self, atoms=None):
        """
        Return the atomic forces.
        """
        forces = np.insert(self.get_property('forces', atoms), 0, 0, 0)
        clf = self.transformer.get_index_transformer(atoms)
        return clf.map_forces(forces, reverse=True)

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

        with self._graph.as_default():
            ops = {target: self._ops[target] for target in properties}
            self.results = self._sess.run(
                ops, feed_dict=self._transformer.get_feed_dict(atoms))
