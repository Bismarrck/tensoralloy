#!coding=utf-8
"""
The GenericRadialAtomicPotential, GRAP
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_estimator import estimator as tf_estimator
from typing import List, Dict, Union
from sklearn.model_selection import ParameterGrid

from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.descriptor.cutoff import cosine_cutoff, polynomial_cutoff
from tensoralloy.precision import get_float_dtype
from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors
from tensoralloy.nn.atomic.dataclasses import FiniteTemperatureOptions
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.eam.potentials.generic import morse, density_exp, power_exp


class Algorithm:
    """
    The base class for all radial descriptors.
    """
    
    required_keys = []
    name = "algorithm"
    
    def __init__(self,
                 parameters: Dict[str, Union[List[float], np.ndarray]],
                 param_space_method="cross"):
        """
        Initialization method.

        Parameters
        ----------
        parameters : dict
            A dict of {param_name: values}.
        param_space_method : str
            The method to build the parameter space, 'cross' or 'pair'. Assume
            there are M types of parameters and each has N_k (1 <= k <= M)
            values, the total number of parameter sets will be:
                * cross: N_1 * N_2 * ... * N_M
                * pair: N_k (1 <= k <= M) must be the same

        """
        assert param_space_method in ("cross", "pair")

        for key in self.required_keys:
            assert key in parameters and len(parameters[key]) >= 1
        self._params = {key: [float(x) for x in parameters[key]] 
                        for key in self.required_keys}
        self._param_space_method = param_space_method

        if param_space_method == "cross":
            self._grid = ParameterGrid(self._params)
        else:
            n = len(set([len(x) for x in self._params.values()]))
            if n > 1:
                raise ValueError(
                    "Hyperparameters must have the same length for gen:pair")
            self._grid = []
            size = len(self._params[self.required_keys[0]])
            for i in range(size):
                row = {}
                for key in self._params.keys():
                    row[key] = self._params[key][i]
                self._grid.append(row)
    
    def __len__(self):
        """
        Return the number of hyper-parameter combinations.
        """
        return len(self._grid)

    def __getitem__(self, item):
        return self._grid[item]
    
    def as_dict(self):
        """
        Return a JSON serializable dict representation of this object.
        """
        return {"algorithm": self.name, "parameters": self._params,
                "param_space_method": self._param_space_method}

    def compute(self, tau: int, rij: tf.Tensor, rc: tf.Tensor,
                dtype=tf.float32):
        """
        Compute f(r, rc) using the \tau-th set of parameters.
        
        Notes
        -----
        The smooth damping `cutoff(r, rc)`, the multipole effect and zero-masks 
        will be handled outside this function.
        
        """
        raise NotImplementedError()


class SymmetryFunctionAlgorithm(Algorithm):
    """
    The radial symmetry function descriptor.
    """
    
    required_keys = ['eta', 'omega']
    name = "sf"
    
    def compute(self, tau: int, rij: tf.Tensor, rc: tf.Tensor,
                dtype=tf.float32):
        """
        Compute f(r, rc) using the \tau-th set of parameters.
        """
        omega = tf.convert_to_tensor(
            self._grid[tau]['omega'], dtype=dtype, name='omega')
        eta = tf.convert_to_tensor(
            self._grid[tau]['eta'], dtype=dtype, name='eta')
        rc2 = tf.multiply(rc, rc, name='rc2')
        z = tf.math.truediv(
            tf.square(tf.math.subtract(rij, omega)), rc2, name='z')
        v = tf.exp(tf.negative(tf.math.multiply(z, eta)))
        return v


class MorseAlgorithm(Algorithm):
    """
    The morse-style descriptor.
    """
    
    required_keys = ['D', 'gamma', 'r0']
    name = "morse"
    
    def compute(self, tau: int, rij: tf.Tensor, rc: tf.Tensor,
                dtype=tf.float32):
        """
        Compute f(r, rc) using the \tau-th set of parameters.
        """
        d = tf.convert_to_tensor(
            self._grid[tau]['D'], dtype=dtype, name='D')
        gamma = tf.convert_to_tensor(
            self._grid[tau]['gamma'], dtype=dtype, name='gamma')
        r0 = tf.convert_to_tensor(
            self._grid[tau]['r0'], dtype=dtype, name='r0')
        return morse(rij, d=d, gamma=gamma, r0=r0)


class DensityExpAlgorithm(Algorithm):
    """
    The expoential density descriptor: f(r) = A * exp[ -beta * (r / re - 1) ]
    """
    
    required_keys = ['A', 'beta', 're']
    name = "density"
    
    def compute(self, tau: int, rij: tf.Tensor, rc: tf.Tensor,
                dtype=tf.float32):
        """
        Compute f(r, rc) using the \tau-th set of parameters.
        """
        a = tf.convert_to_tensor(
            self._grid[tau]['A'], dtype=dtype, name='A')
        beta = tf.convert_to_tensor(
            self._grid[tau]['beta'], dtype=dtype, name='beta')
        re = tf.convert_to_tensor(
            self._grid[tau]['re'], dtype=dtype, name='re')
        return density_exp(rij, a=a, b=beta, re=re)


class PowerExpAlgorithm(Algorithm):
    """
    The power-exponential descriptor used by Oganov.
    """

    required_keys = ["rl", "pl"]
    name = "pexp"

    def compute(self, tau: int, rij: tf.Tensor, rc: tf.Tensor,
                dtype=tf.float32):
        """
        Compute f(r, rc) using the \tau-th set of parameters.
        """
        rl = tf.convert_to_tensor(
            self._grid[tau]['rl'], dtype=dtype, name='rl')
        pl = tf.convert_to_tensor(
            self._grid[tau]['pl'], dtype=dtype, name='pl')
        return power_exp(rij, rl, pl)


class GenericRadialAtomicPotential(AtomicNN):
    """
    The generic atomic potential with polarized radial interactions.
    """

    scope = "GRAP"
    
    def __init__(self,
                 elements: List[str],
                 hidden_sizes=None,
                 activation=None,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces'),
                 kernel_initializer="he_normal",
                 minmax_scale=False,
                 use_atomic_static_energy=True,
                 fixed_atomic_static_energy=False,
                 atomic_static_energy=None,
                 use_resnet_dt=True,
                 finite_temperature=FiniteTemperatureOptions(),
                 algorithm='sf',
                 parameters=None,
                 param_space_method="cross",
                 moment_tensors: Union[int, List[int]] = 0,
                 cutoff_function="cosine"):
        """
        Initialization method.
        """
        super(GenericRadialAtomicPotential, self).__init__(
            elements=elements,
            hidden_sizes=hidden_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_atomic_static_energy=use_atomic_static_energy,
            fixed_atomic_static_energy=fixed_atomic_static_energy,
            atomic_static_energy=atomic_static_energy,
            use_resnet_dt=use_resnet_dt,
            minmax_scale=minmax_scale,
            finite_temperature=finite_temperature,
            minimize_properties=minimize_properties,
            export_properties=export_properties)
        
        if isinstance(moment_tensors, int):
            moment_tensors = [moment_tensors]
        moment_tensors = list(set(moment_tensors))
        for moment_tensor in moment_tensors:
            assert 0 <= moment_tensor <= 2
        
        self._algorithm = self.initialize_algorithm(
            algorithm, parameters, param_space_method)
        self._moment_tensors = moment_tensors
        self._cutoff_function = cutoff_function
    
    def as_dict(self):
        """
        Return a JSON serializable dict representation of GRAP.
        """
        d = super(GenericRadialAtomicPotential, self).as_dict()
        d["moment_tensors"] = self._moment_tensors
        d.update(self._algorithm.as_dict())
        return d

    @staticmethod
    def initialize_algorithm(algorithm: str,
                             parameters: dict,
                             param_space_method='cross'):
        """
        Initialize an `Algorithm` object.
        """
        for cls in (SymmetryFunctionAlgorithm,
                    DensityExpAlgorithm,
                    PowerExpAlgorithm,
                    MorseAlgorithm):
            if cls.name == algorithm:
                break
        else:
            raise ValueError(
                f"GRAP: algorithm '{algorithm}' is not implemented")
        return cls(parameters, param_space_method)
   
    def apply_cutoff(self, x, rc, name=None):
        """
        Apply the cutoff function on interatomic distances.
        """
        if self._cutoff_function == "cosine":
            return cosine_cutoff(x, rc, name=name)
        else:
            return polynomial_cutoff(x, rc, name=name)
    
    def apply_pairwise_descriptor_functions(self, partitions: dict):
        """
        Apply the descriptor functions to all partitions.
        """
        xyz_map = {
            0: 'x', 1: 'y', 2: 'z'
        }
        moment_tensors_indices = {
            1: [0, 1, 2],
            2: [(0, 1), (0, 2), (1, 2)]
        }
        
        clf = self._transformer
        dtype = get_float_dtype()
        rc = tf.convert_to_tensor(clf.rcut, name='rc', dtype=dtype)
        outputs = {element: [None] * len(self._elements)
                   for element in self._elements}
        for kbody_term, (dists, masks) in partitions.items():
            center = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"{kbody_term}"):
                rij = tf.squeeze(dists[0], axis=1, name='rij')
                dij = tf.squeeze(dists[1:], axis=2, name='dij')
                masks = tf.squeeze(masks, axis=1, name='masks')
                fc = self.apply_cutoff(rij, rc=rc, name='fc')
                gtau = []
                
                def _post_compute(fx):
                    """
                    Apply the smooth cutoff values and zero masks to `fx`.
                    Then sum `fx` for each atom.
                    """
                    gx = tf.math.multiply(fx, fc)
                    gx = tf.math.multiply(gx, masks, name=f'gx/masked')
                    gx = tf.expand_dims(
                        tf.reduce_sum(gx, axis=[-1, -2], keep_dims=False),
                        axis=-1, name='gx')
                    return gx
                
                for tau in range(len(self._algorithm)):
                    with tf.name_scope(f"{tau}"):
                        v = self._algorithm.compute(
                            tau, rij, rc, dtype=dtype)
                        # The standard central-force model
                        if 0 in self._moment_tensors:
                            with tf.name_scope("r"):
                                gtau.append(_post_compute(v))
                        # Dipole effect
                        if 1 in self._moment_tensors:
                            for i in moment_tensors_indices[1]:
                                itag = xyz_map[i]
                                with tf.name_scope(f"r{itag}"):
                                    coef = tf.div_no_nan(
                                        dij[i], rij, name='coef')
                                    gtau.append(
                                        _post_compute(
                                            tf.multiply(v, coef, name='fx')))
                        # Quadrupole effect
                        if 2 in self._moment_tensors:
                            for (i, j) in moment_tensors_indices[2]:
                                itag = xyz_map[i]
                                jtag = xyz_map[j]
                                with tf.name_scope(f"r{itag}{jtag}"):
                                    coef = tf.div_no_nan(
                                        dij[i] * dij[j], rij * rij, name='coef')
                                    gtau.append(
                                        _post_compute(
                                            tf.multiply(v, coef, name='fx')))
                g = tf.concat(gtau, axis=-1, name='g')
            index = clf.kbody_terms_for_element[center].index(kbody_term)
            outputs[center][index] = g
        with tf.name_scope("Concat"):
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    outputs[element], axis=-1, name=element)
            return results

    def _get_atomic_descriptors(self,
                                universal_descriptors,
                                mode: tf_estimator.ModeKeys,
                                verbose=True):
        """
        Construct the computation graph for calculating descriptors.
        """
        clf = self._transformer
        with tf.name_scope("Radial"):
            partitions, max_occurs = dynamic_partition(
                dists_and_masks=universal_descriptors['radial'],
                elements=clf.elements,
                kbody_terms_for_element=clf.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=False)
            descriptors = self.apply_pairwise_descriptor_functions(partitions)
        return AtomicDescriptors(descriptors=descriptors, max_occurs=max_occurs)
