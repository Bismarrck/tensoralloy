!#coding=utf-8
from __future__ import print_function

import tensorflow as tf

from typing import List, Dict
from sklearn.model_selection import ParameterGrid

from tensoralloy.nn.atomic.atomic import AtomicNN
from tensoralloy.nn.eam.potentials.generic import morse, density_exp

available_algorithms = ['sf', 'morse', 'density']


class Algorithm:
    """
    The base class for all radial descriptors.
    """
    
    required_keys = []
    name = "algorithm"
    
    def __init__(self, parameters: Dict[str, Union[List[float], np.ndarray]]):
        """
        Initialization method.
        """
        for key in required_keys:
            assert key in parameters and len(parameters[key]) >= 1
        self._params = {key: [float(x) for x in parameters[key]] 
                        for key in required_keys}
        self._grid = ParameterGrid(**self._params)
    
    def __len__(self):
        return len(self._grid)
    
    def as_dict(self):
        """
        Return a JSON serializable dict representation of this object.
        """
        return {"algorithm": self.name, "parameters": self._params}
    
    def compute(self, tau: int, rij: tf.Tensor, dtype=tf.float32):
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
    
    def compute(self, tau: int, rij: tf.Tensor, dtype=tf.float32):
        """
        Compute f(r, rc) using the \tau-th set of parameters.
        """
        omega = tf.convert_to_tensor(
            self._grid[tau]['omega'], dtype=dtype, name='omega')
        eta = tf.convert_to_tensor(
            self._grid[tau]['eta'], dtype=dtype, name='eta')
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
    
    def compute(self, tau: int, rij: tf.Tensor, dtype=tf.float32):
        """
        Compute f(r, rc) using the \tau-th set of parameters.
        """
        d = tf.convert_to_tensor(
            self._grid[tau]['D'], dtype=dtype, name='D')
        gamma = tf.convert_to_tensor(
            self._grid[tau]['gamma'], dtype=dtype, name='gamma')
        r0 = tf.convert_to_tensor(
            self._grid[tau]['r0'], dtype=dtype, name='r0')
        return morse(rij, D=d, gamma=gamma, r0=r0)


class DensityExpAlgorithm(Algorithm):
    """
    The expoential density descriptor: f(r) = A * exp[ -beta * (r / re - 1) ]
    """
    
    required_keys = ['A', 'beta', 're']
    name = "density
    
    def compute(self, tau: int, rij: tf.Tensor, dtype=tf.float32):
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
    

class GenericRadialAtomicPotential(AtomicNN):
    
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
                 temperature_dependent=False,
                 temperature_layers=(128, 128),
                 temperature_activation='softplus',
                 algorithm='sf',
                 parameters=None,
                 multipole=0,
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
            temperature_dependent=temperature_dependent,
            temperature_layers=temperature_layers,
            temperature_activation=temperature_activation,
            minimize_properties=minimize_properties,
            export_properties=export_properties)
        
        if algorithm not in available_algorithms:
            raise ValueError(f"GRAP: algorithm '{algorithm}' is not implemented")
        
        self._algorithm = initialize_algorithm(algorithm, parameters)
        self._multipole = multipole
        self._cutoff_function = cutoff_function
        self._nn_scope = "Atomic/GRAP"
    
    def as_dict(self):
        """
        Return a JSON serializable dict representation of GRAP.
        """
        d = super(GenericRadialAtomicPotential, self).as_dict()
        d["multipole"] = self._multipole
        d.update(self._algorithm.as_dict())
        return d
    
    def initialize_algorithm(self, algorithm: str, parameters: dict):
        """
        Initialize an `Algorithm` object.
        """
        if self._algorithm == "sf":
            cls = SymmetryFunctionAlgorithm
        elif self._algorithm == "density":
            cls = DensityExpAlgorithm
        else:
            cls = MorseAlgorithm
        return cls(parameters)
   
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
        clf = self._transformer
        dtype = get_float_dtype()
        rc = tf.convert_to_tensor(clf.rcut, name='rc', dtype=dtype)
        rc2 = tf.convert_to_tensor(clf.rcut**2, dtype=dtype, name='rc2')
        outputs = {element: [None] * len(self._elements)
                   for element in self._elements}
        for kbody_term, (dists, masks) in partitions.items():
            center = get_elements_from_kbody_term(kbody_term)[0]
            with tf.variable_scope(f"{kbody_term}"):
                rij = tf.squeeze(dists[0], axis=1, name='rij')
                ijx = tf.squeeze(dists[1], axis=1, name='ijx')
                ijy = tf.squeeze(dists[2], axis=1, name='ijy')
                ijz = tf.squeeze(dists[3], axis=1, name='ijz')
                masks = tf.squeeze(masks, axis=1, name='masks')
                fc = self.apply_cutoff(rij, rc=rc, name='fc')
                gtau = []
                for tau in range(len(self._algorithm)):
                    with tf.name_scope(f"{tau}"):
                        v = self._algorithm.compute(tau, rij, dtype=dtype)
                        v = tf.math.multiply(v, fc)
                        v = tf.math.multiply(v, masks, name='v/masked')
                        g = tf.expand_dims(
                            tf.reduce_sum(v, axis=[-1, -2], keep_dims=False),
                            axis=-1, name='g')
                        gtau.append(g)
                g = tf.concat(gtau, axis=-1, name='g')
            index = clf.kbody_terms_for_element[center].index(kbody_term)
            outputs[center][index] = g
        with tf.name_scope("Concat"):
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    outputs[element], axis=-1, name=element)
            return results
    
