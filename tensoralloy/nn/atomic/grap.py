#!coding=utf-8
"""
The GenericRadialAtomicPotential, GRAP
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf

from collections import Counter
from typing import List, Dict, Union
from sklearn.model_selection import ParameterGrid

from tensoralloy.transformer import UniversalTransformer
from tensoralloy.utils import get_elements_from_kbody_term, ModeKeys
from tensoralloy.precision import get_float_dtype
from tensoralloy.nn.cutoff import cosine_cutoff, polynomial_cutoff
from tensoralloy.nn.atomic.atomic import Descriptor
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.eam.potentials.generic import morse, density_exp, power_exp
from tensoralloy.nn.eam.potentials.generic import power_exp1, power_exp2, power_exp3


GRAP_algorithms = ["pexp", "density", "morse", "sf"]


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
        pl = self._grid[tau]['pl']
        if pl == 1.0:
            return power_exp1(rij, rl)
        elif pl == 2.0:
            return power_exp2(rij, rl)
        elif pl == 3.0:
            return power_exp3(rij, rl)
        else:
            pl = tf.convert_to_tensor(pl, dtype=dtype, name='pl')
            return power_exp(rij, rl, pl)


class GenericRadialAtomicPotential(Descriptor):
    """
    The generic atomic potential with polarized radial interactions.
    """

    def __init__(self,
                 elements: List[str],
                 algorithm='sf',
                 parameters=None,
                 param_space_method="pair",
                 moment_tensors: Union[int, List[int]] = 0,
                 moment_scale_factors: Union[float, List[float]] = 1.0,
                 cutoff_function="cosine",
                 legacy_mode=True):
        """
        Initialization method.
        """
        super(GenericRadialAtomicPotential, self).__init__(elements=elements)
        
        if isinstance(moment_tensors, int):
            moment_tensors = [moment_tensors]
        moment_tensors = list(set(moment_tensors))
        if np.isscalar(moment_scale_factors):
            moment_scale_factors = [moment_scale_factors] * len(moment_tensors)
        else:
            assert len(moment_scale_factors) == len(moment_tensors)

        self._algorithm = algorithm
        self._algorithm_instance = self.initialize_algorithm(
            algorithm, parameters, param_space_method)
        self._moment_tensors = moment_tensors
        self._moment_scale_factors = moment_scale_factors
        self._cutoff_function = cutoff_function
        self._parameters = parameters
        self._param_space_method = param_space_method
        self._legacy_mode = legacy_mode

    @property
    def name(self):
        """ Return the name of this descriptor. """
        return "GRAP"

    def as_dict(self):
        """
        Return a JSON serializable dict representation of GRAP.
        """
        d = super(GenericRadialAtomicPotential, self).as_dict()
        d.update({"moment_tensors": self._moment_tensors,
                  "cutoff_function": self._cutoff_function,
                  "moment_scale_factors": self._moment_scale_factors,
                  "legacy_mode": self._legacy_mode})
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
        results : Dict
            A dict. The keys are elements and the values are corresponding
            merged features.

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
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    stacks[element], axis=-1, name=element)
            return results
    
    def apply_symmetry_preserved_model(self, 
                                       clf: UniversalTransformer, 
                                       partitions: dict, 
                                       max_occurs: Counter):
        """
        Apply the symmetry preserved model.
        """
        dtype = get_float_dtype()
        rc = tf.convert_to_tensor(clf.rcut, name='rc', dtype=dtype)
        outputs = {}
        m1 = len(self._algorithm_instance)
        m2 = min(m1, abs(self._moment_tensors[0]))
        for kbody_term, (dists, masks) in partitions.items():
            with tf.variable_scope(f"{kbody_term}"):
                ijx = tf.squeeze(dists[1], axis=1, name='ijx')
                ijy = tf.squeeze(dists[2], axis=1, name='ijy')
                ijz = tf.squeeze(dists[3], axis=1, name='ijz')
                rij = tf.squeeze(dists[0], axis=1, name='rij')
                sij = self.apply_cutoff(rij, rc=rc, name='sij')
                zij = tf.div_no_nan(sij, rij, name="zij")
                ijx = tf.multiply(ijx, zij, name="ijx/hat")
                ijy = tf.multiply(ijy, zij, name="ijy/hat")
                ijz = tf.multiply(ijz, zij, name="ijz/hat")
                masks = tf.squeeze(masks, axis=1, name='masks')
                gtau = []
                for tau in range(m1):
                    with tf.name_scope(f"{tau}"):
                        v = self._algorithm_instance.compute(
                            tau, rij, rc, dtype=dtype)
                        gtau.append(v)
                g = tf.concat(gtau, axis=-1, name="G")
                g1 = tf.multiply(g, masks, name="G1")
                g2 = tf.identity(g1[..., :m2], name="G2")
                rr = tf.concat((sij, ijx, ijy, ijz), axis=-1, name='R')
                d1 = tf.einsum('ijkl,ijkp->ijlp', g1, rr, name='GR')
                d2 = tf.einsum('ijkl,ijpl->ijkp', d1, rr, name='GRR')
                d3 = tf.einsum('ijkl,ijlp->ijkp', d2, g2, name='GRRG')
                shape = tf.shape(d3)
                x = tf.reshape(d3, [shape[0], shape[1], m1 * m2],
                               name='D')
                outputs[kbody_term] = x
        return self._dynamic_stitch(outputs, max_occurs, False)
   
    def apply_cutoff(self, x, rc, name=None):
        """
        Apply the cutoff function on interatomic distances.
        """
        if self._cutoff_function == "cosine":
            return cosine_cutoff(x, rc, name=name)
        else:
            return polynomial_cutoff(x, rc, name=name)

    def apply_pairwise_descriptor_functions(self,
                                            clf: UniversalTransformer,
                                            partitions: dict):
        """
        Apply the descriptor functions to all partitions.
        """
        xyz_map = {
            0: 'x', 1: 'y', 2: 'z'
        }
        vind = {
            1: {0: 1, 1: 1, 2: 1},
            2: {(0, 0): 1, (1, 1): 1, (2, 2): 1,
                (0, 1): 2, (0, 2): 2, (1, 2): 2},
            3: {(0, 0, 0): 1, (1, 1, 1): 1, (2, 2, 2): 1,
                (0, 0, 1): 3, (0, 0, 2): 3, (0, 1, 1): 3, (0, 1, 2): 6,
                (0, 2, 2): 3, (1, 1, 2): 3, (1, 2, 2): 3}}
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
                c2 = tf.constant(1.0 / 3.0, dtype=dtype, name="c2")
                c3 = tf.constant(3.0 / 5.0, dtype=dtype, name="c3")
                gtau = []

                def compute(fx, square=False):
                    """
                    Apply the smooth cutoff values and zero masks to `fx`.
                    Then sum `fx` for each atom.
                    """
                    gx = tf.math.multiply(fx, fc)
                    gx = tf.math.multiply(gx, masks, name=f'gx/masked')
                    gx = tf.expand_dims(
                        tf.reduce_sum(gx, axis=[-1, -2], keep_dims=False),
                        axis=-1, name='gx')
                    if square:
                        gx = tf.square(gx, name='gx2')
                    return gx

                for tau in range(len(self._algorithm_instance)):
                    with tf.name_scope(f"{tau}"):
                        v = self._algorithm_instance.compute(
                            tau, rij, rc, dtype=dtype)
                        squared = {}
                        for idx, angular in enumerate(self._moment_tensors):
                            vtau = []
                            if angular == 0:
                                with tf.name_scope("r"):
                                    g = compute(v)
                                    g2 = tf.square(g, name="g/squared")
                                    gtau.append(g)
                                    squared[0] = g2
                            elif angular == 1:
                                for i in vind[1]:
                                    itag = xyz_map[i]
                                    with tf.name_scope(f"r{itag}"):
                                        coef = tf.div_no_nan(
                                            dij[i], rij, name='coef')
                                        vi = compute(
                                            tf.multiply(v, coef, name='fx'),
                                            square=True)
                                        vtau.append(vi)
                                g2 = tf.add_n(vtau, name="g/squared")
                                g = tf.sqrt(g2, name="g")
                                squared[1] = g2
                                gtau.append(g)
                            elif angular == 2:
                                for (i, j), multi in vind[2].items():
                                    itag = xyz_map[i]
                                    jtag = xyz_map[j]
                                    with tf.name_scope(f"r{itag}{jtag}"):
                                        coef = tf.div_no_nan(
                                            dij[i] * dij[j], rij * rij,
                                            name='coef')
                                        v = compute(
                                            tf.multiply(v, coef, name='fx'))
                                        multi = tf.convert_to_tensor(
                                            multi, dtype=dtype, name="multi")
                                        vtau.append(tf.square(v) * multi)
                                g2 = tf.subtract(tf.add_n(vtau),
                                                 squared[0] * c2,
                                                 name="g/squared")
                                g = tf.sqrt(g2, name="g")
                                gtau.append(g)
                            elif angular == 3:
                                for (i, j, k), multi in vind[3].items():
                                    itag = xyz_map[i]
                                    jtag = xyz_map[j]
                                    ktag = xyz_map[k]
                                    with tf.name_scope(f"r{itag}{jtag}{ktag}"):
                                        coef = tf.div_no_nan(
                                            dij[i] * dij[j] * dij[k],
                                            rij * rij * rij,
                                            name='coef')
                                        v = compute(
                                            tf.multiply(v, coef, name='fx'))
                                        multi = tf.convert_to_tensor(
                                            multi, dtype=dtype, name="multi")
                                        vtau.append(tf.square(v) * multi)
                                g2 = tf.subtract(tf.add_n(vtau),
                                                 squared[1] * c3,
                                                 name="g/squared")
                                g = tf.sqrt(g2, name="g")
                                gtau.append(g)
                            else:
                                raise ValueError(
                                    "Angular moment should be <= 3")
                g = tf.concat(gtau, axis=-1, name='g')
            index = clf.kbody_terms_for_element[center].index(kbody_term)
            outputs[center][index] = g
        with tf.name_scope("Concat"):
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    outputs[element], axis=-1, name=element)
            return results

    def apply_legacy_pairwise_descriptor_functions(self,
                                                   clf: UniversalTransformer,
                                                   partitions: dict):
        """
        Apply the descriptor functions to all partitions.
        """
        xyz_map = {
            0: 'x', 1: 'y', 2: 'z'
        }
        moment_tensors_indices = {1: [0, 1, 2],
                                  2: [(0, 0), (1, 1), (2, 2),
                                      (0, 1), (0, 2), (1, 2),
                                      (1, 0), (2, 0), (2, 1)]}
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
                
                def compute(fx, square=False):
                    """
                    Apply the smooth cutoff values and zero masks to `fx`.
                    Then sum `fx` for each atom.
                    """
                    gx = tf.math.multiply(fx, fc)
                    gx = tf.math.multiply(gx, masks, name=f'gx/masked')
                    gx = tf.expand_dims(
                        tf.reduce_sum(gx, axis=[-1, -2], keep_dims=False),
                        axis=-1, name='gx')
                    if square:
                        gx = tf.square(gx, name='gx2')
                    return gx
                
                for tau in range(len(self._algorithm_instance)):
                    with tf.name_scope(f"{tau}"):
                        v = self._algorithm_instance.compute(
                            tau, rij, rc, dtype=dtype)
                        for idx, moment_tensor in \
                                enumerate(self._moment_tensors):
                            scale = tf.constant(
                                self._moment_scale_factors[idx],
                                name=f'factor{idx}',
                                dtype=dtype)
                            if moment_tensor == 0:
                                # The standard central-force part
                                with tf.name_scope("r"):
                                    gtau.append(compute(v))
                            elif moment_tensor == 1:
                                # Dipole moment
                                dtau = []
                                for i in moment_tensors_indices[1]:
                                    itag = xyz_map[i]
                                    with tf.name_scope(f"r{itag}"):
                                        coef = tf.div_no_nan(
                                            dij[i], rij, name='coef')
                                        vi = compute(
                                            tf.multiply(v, coef, name='fx'),
                                            square=True)
                                        dtau.append(vi)
                                gtau.append(tf.multiply(scale, tf.add_n(dtau),
                                                        name='u2'))
                            elif moment_tensor in [2, 21]:
                                # Quadrupole moment
                                qtau = []
                                vtau = []
                                for (i, j) in moment_tensors_indices[2]:
                                    itag = xyz_map[i]
                                    jtag = xyz_map[j]
                                    with tf.name_scope(f"r{itag}{jtag}"):
                                        coef = tf.div_no_nan(
                                            dij[i] * dij[j], rij * rij,
                                            name='coef')
                                        vij = compute(
                                            tf.multiply(v, coef, name='fx'))
                                        if i == j:
                                            vtau.append(vij)
                                        qtau.append(tf.square(vij))
                                gtau.append(tf.multiply(scale, tf.add_n(qtau),
                                                        name='q2'))
                                if moment_tensor == 21:
                                    gtau.append(tf.multiply(
                                        scale, tf.square(tf.add_n(vtau)),
                                        name='v2'))
                g = tf.concat(gtau, axis=-1, name='g')
            index = clf.kbody_terms_for_element[center].index(kbody_term)
            outputs[center][index] = g
        with tf.name_scope("Concat"):
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    outputs[element], axis=-1, name=element)
            return results

    def calculate(self,
                  transformer: UniversalTransformer,
                  universal_descriptors,
                  mode: ModeKeys,
                  verbose=False) -> AtomicDescriptors:
        """
        Construct the computation graph for calculating descriptors.
        """
        with tf.name_scope("Radial"):
            partitions, max_occurs = dynamic_partition(
                dists_and_masks=universal_descriptors['radial'],
                elements=transformer.elements,
                kbody_terms_for_element=transformer.kbody_terms_for_element,
                mode=mode,
                angular=False,
                merge_symmetric=False)
            if len(self._moment_tensors) == 1 and self._moment_tensors[0] < 0:
                g = self.apply_symmetry_preserved_model(
                    transformer, partitions, max_occurs)
            else:
                if self._legacy_mode:
                    g = self.apply_legacy_pairwise_descriptor_functions(
                        transformer, partitions)
                else:
                    g = self.apply_pairwise_descriptor_functions(
                        transformer, partitions)
        return AtomicDescriptors(descriptors=g, max_occurs=max_occurs)
