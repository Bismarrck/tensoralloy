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
from ase.data import covalent_radii, atomic_numbers

from tensoralloy.transformer import UniversalTransformer
from tensoralloy.utils import get_elements_from_kbody_term, ModeKeys
from tensoralloy.nn.convolutional import convolution1x1
from tensoralloy.precision import get_float_dtype
from tensoralloy.nn.cutoff import cosine_cutoff, polynomial_cutoff
from tensoralloy.nn.atomic.atomic import Descriptor
from tensoralloy.nn.atomic.dataclasses import AtomicDescriptors
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.eam.potentials.generic import morse, density_exp, power_exp
from tensoralloy.nn.eam.potentials.generic import power_exp1, power_exp2
from tensoralloy.nn.eam.potentials.generic import power_exp3
from tensoralloy.nn.utils import get_activation_fn

GRAP_algorithms = ["pexp", "density", "morse", "sf", "nn"]


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

    def as_dict(self, convert_to_pairs=False):
        """
        Return a JSON serializable dict representation of this object.
        """
        if not convert_to_pairs:
            parameters = {}
            for key, value in self._params.items():
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                parameters[key] = value
            param_space_method = self._param_space_method
        else:
            param_space_method = "pair"
            parameters = {}
            for key in self._params:
                parameters[key] = []
            for i in range(len(self._grid)):
                for key in self._params:
                    parameters[key].append(float(self._grid[i][key]))
        return {"algorithm": self.name, "parameters": parameters,
                "param_space_method": param_space_method}

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


class NNAlgorithm:
    """
    The Neural Network descriptor model.
    """

    required_keys = ["activation_fn", "hidden_sizes", "num_filters",
                     "use_reset_dt", "ckpt", "trainable", "h_abck_modifier"]
    name = "nn"

    def __init__(self, parameters: dict):
        self.use_resnet_dt = parameters.get("use_resnet_dt", True)
        self.hidden_sizes = parameters.get("hidden_sizes", [32, 32, 32])
        self.activation = parameters.get("activation", "softplus")
        self.num_filters = parameters.get("num_filters", 16)
        self.ckpt = parameters.get("ckpt", None)
        self.trainable = parameters.get("trainable", True)
        self.h_abck_modifier = parameters.get("h_abck_modifier", 0)

    def __len__(self):
        return self.num_filters

    def __getitem__(self, item):
        return self.__dict__[item]

    def as_dict(self, convert_to_pairs=False):
        """
        Dict representation of this class.
        """
        if isinstance(self.ckpt, str):
            npz = np.load(self.ckpt)
            actfn_map = {
                0: "relu",
                1: "softplus",
                2: "tanh",
                3: "squareplus",
            }
            self.activation = actfn_map.get(int(npz["fnn::actfn"]))
            self.hidden_sizes = npz["fnn::layer_sizes"].tolist()
            self.num_filters = self.hidden_sizes.pop(-1)
            self.use_resnet_dt = npz["fnn::use_resnet_dt"] == 1
            self.trainable = npz["fnn::trainable"] == 1
            self.h_abck_modifier = npz.get("fnn::h_abck_modifier", 0)

        return {"use_resnet_dt": self.use_resnet_dt,
                "hidden_sizes": self.hidden_sizes,
                "activation": self.activation,
                "num_filters": self.num_filters,
                "trainable": self.trainable,
                "ckpt": self.ckpt,
                "h_abck_modifier": self.h_abck_modifier}


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
                 cutoff_function="cosine",
                 symmetric=False,
                 legacy_mode=True,
                 h_abck_modifier=None):
        """
        Initialization method.
        """
        super(GenericRadialAtomicPotential, self).__init__(elements=elements)

        if isinstance(moment_tensors, int):
            moment_tensors = [moment_tensors]
        moment_tensors = list(set(moment_tensors))

        if algorithm == "nn":
            if legacy_mode:
                raise ValueError(
                    "The NN algorithm cannot be used for GRAP legacy mode")
            self._use_nn = True
            self._algo = NNAlgorithm(parameters)
        else:
            self._use_nn = False
            self._algo = self.initialize_algorithm(
                algorithm, parameters, param_space_method)

        self._moment_tensors = moment_tensors
        self._cutoff_function = cutoff_function
        self._parameters = parameters
        self._param_space_method = param_space_method
        self._legacy_mode = legacy_mode
        self._symmetric = symmetric

    @property
    def name(self):
        """ Return the name of this descriptor. """
        return "GRAP"

    @property
    def algorithm(self):
        """ The descriptor model. """
        return self._algo

    @property
    def max_moment(self):
        """ The maximum angular moment. """
        return max(self._moment_tensors)

    @property
    def is_T_symmetric(self):
        """
        Is the multiplicity `T_md` tensor symmetric?
        """
        return self._symmetric

    @property
    def cutoff_function(self):
        """ The cutoff function. """
        return self._cutoff_function

    def as_dict(self):
        """
        Return a JSON serializable dict representation of GRAP.
        """
        d = {"@class": self.__class__.__name__,
             "@module": self.__class__.__module__,
             "elements": self._elements,
             "algorithm": self._algo.name,
             "parameters": self._parameters,
             "param_space_method": self._param_space_method,
             "moment_tensors": self._moment_tensors,
             "cutoff_function": self._cutoff_function,
             "symmetric": self._symmetric,
             "legacy_mode": self._legacy_mode}
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

                for tau in range(len(self._algo)):
                    with tf.name_scope(f"{tau}"):
                        v = self._algo.compute(
                            tau, rij, rc, dtype=dtype)
                        for moment in self._moment_tensors:
                            if moment == 0:
                                # The standard central-force part
                                with tf.name_scope("r"):
                                    gtau.append(compute(v))
                            elif moment == 1:
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
                                gtau.append(tf.add_n(dtau, name='u2'))
                            elif moment == 2:
                                # Quadrupole moment
                                qtau = []
                                for (i, j) in moment_tensors_indices[2]:
                                    itag = xyz_map[i]
                                    jtag = xyz_map[j]
                                    with tf.name_scope(f"r{itag}{jtag}"):
                                        coef = tf.div_no_nan(
                                            dij[i] * dij[j], rij * rij,
                                            name='coef')
                                        vij = compute(
                                            tf.multiply(v, coef, name='fx'))
                                        qtau.append(tf.square(vij))
                                gtau.append(tf.add_n(qtau, name='q2'))
                g = tf.concat(gtau, axis=-1, name='g')
            index = clf.kbody_terms_for_element[center].index(kbody_term)
            outputs[center][index] = g
        with tf.name_scope("Concat"):
            results = {}
            for element in self._elements:
                results[element] = tf.concat(
                    outputs[element], axis=-1, name=element)
            return results

    @staticmethod
    def _get_multiplicity_tensor(max_moment: int, symmetric=False):
        """
        Return the multiplicity tensor T_dm.
        """
        dtype = get_float_dtype()
        if max_moment == 0:
            array = np.ones((1, 1), dtype=dtype.as_numpy_dtype)
        elif max_moment == 1:
            array = np.zeros((4, 2), dtype=dtype.as_numpy_dtype)
            array[0, 0] = array[1: 4, 1] = 1
        elif max_moment == 2:
            array = np.zeros((10, 3), dtype=dtype.as_numpy_dtype)
            array[0, 0] = array[1: 4, 1] = 1
            array[4: 10, 2] = 1, 2, 2, 1, 2, 1
            if symmetric:
                array[0, 2] = -1 / 3
        else:
            array = np.zeros((20, 4), dtype=dtype.as_numpy_dtype)
            array[0, 0] = array[1: 4, 1] = 1
            array[4: 10, 2] = 1, 2, 2, 1, 2, 1
            array[10: 20, 3] = 1, 3, 3, 3, 6, 3, 1, 3, 3, 1
            if symmetric:
                array[0, 2] = -1 / 3
                array[1: 4, 3] = -3 / 5
        return tf.convert_to_tensor(array, dtype=dtype, name="T_dm")

    @staticmethod
    def _get_moment_coeff_tensor(rij: tf.Tensor, dij: tf.Tensor,
                                 max_moment: int):
        """
        Return the moment coefficients tensor.
        """
        dtype = get_float_dtype()
        with tf.name_scope("M_dnac"):
            abx = [0, 0, 0, 1, 1, 2]
            aby = [0, 1, 2, 1, 2, 2]
            ab_bcast = tf.convert_to_tensor(
                [9, 1, 1, 1, 1, 1], dtype=tf.int32, name="bcast/ab")
            ab_rows = [abx[i] * 3 + aby[i] for i in range(len(abx))]
            abcx = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2]
            abcy = [0, 0, 0, 1, 1, 2, 1, 1, 2, 2]
            abcz = [0, 1, 2, 1, 2, 2, 1, 2, 2, 2]
            abc_rows = [abcx[i] * 3 ** 2 + abcy[i] * 3 + abcz[i]
                        for i in range(len(abcx))]
            abc_bcast = tf.convert_to_tensor(
                [27, 1, 1, 1, 1, 1], dtype=tf.int32, name="bcast/abc")
            shape = tf.shape(rij, name="shape", out_type=tf.int32)
            M_d = [tf.ones_like(rij, name="d/0", dtype=dtype)]
            za = tf.div_no_nan(dij, rij, name="za")
            if max_moment > 0:
                M_d.append(za)
                if max_moment > 1:
                    xynac = tf.einsum("xnbacu, ynbacu->xynbacu", za, za)
                    zab_flat = tf.reshape(
                        xynac, shape=shape * ab_bcast, name="zab/flat")
                    zab = tf.gather(zab_flat, ab_rows, name="zab")
                    M_d.append(zab)
                    if max_moment > 2:
                        xyznac = tf.einsum("xynbacu, znbacu->xyznbacu",
                                           xynac, za)
                        zabc_flat = tf.reshape(
                            xyznac, shape=shape * abc_bcast, name="zabc/flat")
                        zabc = tf.gather(zabc_flat, abc_rows, name="zabc")
                        M_d.append(zabc)
            return tf.squeeze(tf.concat(M_d, axis=0), axis=-1, name="M")
    
    @staticmethod
    def get_moment_tensor(rij: tf.Tensor, dij: tf.Tensor, max_moment: int):
        """
        The new algorithm to calculate the moment tensor.
        """
        dmax = {0: 1, 1: 4, 2: 13, 3: 40, 4: 121, 5: 364}[max_moment]
        dtype = get_float_dtype()
        with tf.name_scope("M_dnac"):
            shape = tf.shape(rij, name="shape", out_type=tf.int32)
            M_d = [tf.ones_like(rij, name="m/0", dtype=dtype)]
            m_1 = tf.div_no_nan(dij, rij, name="m/1")
            if max_moment > 0:
                M_d.append(m_1)
                if max_moment > 1:
                    m_2 = tf.einsum("xnbacu, ynbacu->xynbacu", m_1, m_1)
                    s_2 = tf.convert_to_tensor([9, 1, 1, 1, 1, 1], dtype=tf.int32, 
                                               name="shape/2")
                    m_2 = tf.reshape(m_2, shape=shape * s_2, name="m/2")
                    M_d.append(m_2)
                    if max_moment > 2:
                        m_3 = tf.einsum("xnbacu, ynbacu->xynbacu", m_2, m_1)
                        s_3 = tf.convert_to_tensor(
                            [27, 1, 1, 1, 1, 1], dtype=tf.int32, name="shape/3")
                        m_3 = tf.reshape(m_3, shape=shape * s_3, name="m/3")
                        M_d.append(m_3)
                        if max_moment > 3:
                            m_4 = tf.einsum("xnbacu, ynbacu->xynbacu", m_3, m_1)
                            s_4 = tf.convert_to_tensor(
                                [81, 1, 1, 1, 1, 1], dtype=tf.int32, name="shape/4")
                            m_4 = tf.reshape(m_4, shape=shape * s_4, name="m/4")
                            M_d.append(m_4)
                            if max_moment > 4:
                                m_5 = tf.einsum("xnbacu, ynbacu->xynbacu", m_4, m_1)
                                s_5 = tf.convert_to_tensor(
                                    [243, 1, 1, 1, 1, 1], dtype=tf.int32, name="s/5")
                                m_5 = tf.reshape(m_5, shape=shape * s_5, name="m_5")
                                M_d.append(m_5)
            return tf.squeeze(tf.concat(M_d, axis=0), axis=-1, name="M") 
    
    @staticmethod
    def get_T_dm(max_moment: int):
        dims = [1, 4, 13, 40, 121, 364]
        if max_moment > 5:
            raise ValueError("The maximum angular moment should be <= 5")
        dmax = dims[max_moment]
        array = np.zeros((dmax, max_moment + 1), dtype=np.float64)
        array[0, 0] = 1
        if max_moment > 0:
            array[1: 4, 1] = 1
        if max_moment > 1:
            array[4: 13, 2] = 1
        if max_moment > 2:
            array[13: 40, 3] = 1
        if max_moment > 3:
            array[40: 121, 4] = 1
        if max_moment > 4:
            array[121: 364, 5] = 1
        return tf.convert_to_tensor(array, dtype=get_float_dtype(), name="T_dm")

    def apply_model(self, clf: UniversalTransformer, dists_and_masks: dict,
                    mode: ModeKeys):
        """
        Apply the descriptor model.
        """
        dtype = get_float_dtype()
        rc = tf.convert_to_tensor(clf.rcut, name='rc', dtype=dtype)
        neltypes = len(self._elements)
        outputs = {element: [None] * len(self._elements)
                   for element in self._elements}
        max_moment = max(self._moment_tensors)
        ndims = (max_moment + 1) * len(self._algo) * neltypes
        max_occurs = Counter()
        for element in clf.elements:
            with tf.name_scope(f"{element}"):
                dists, masks = dists_and_masks[element]
                rij = tf.identity(dists[0], name="rij")
                dij = tf.identity(dists[1:], name="dij")
                if ModeKeys.for_prediction(mode):
                    rij = tf.expand_dims(rij, 0)
                    dij = tf.expand_dims(dij, 1)
                fc = self.apply_cutoff(rij, rc=rc, name='fc')
                fc = tf.multiply(fc, masks, name="fc/masked")
                eps = tf.convert_to_tensor(1e-16, dtype=dtype, name="eps")
                if isinstance(self._algo, NNAlgorithm):
                    if self._algo.h_abck_modifier == 0:
                        h_in = rij
                    elif self._algo.h_abck_modifier == 1:
                        rcov = covalent_radii[atomic_numbers[element]]
                        h_in = tf.divide(rij, rcov, name="h_in/linear")
                    elif self._algo.h_abck_modifier == 2:
                        rcov = covalent_radii[atomic_numbers[element]]
                        h_in = tf.exp(tf.negative(rij / rcov), name="h_in/exp")
                    else:
                        raise ValueError(
                            f"Unknown H(r) modifier: {self._h_abck_modifier}")
                    H = convolution1x1(
                        h_in,
                        activation_fn=get_activation_fn(self._algo.activation),
                        hidden_sizes=self._algo.hidden_sizes,
                        num_out=len(self._algo),
                        variable_scope="Filters",
                        output_bias=False,
                        verbose=True,
                        trainable=self._algo.trainable,
                        ckpt=self._algo.ckpt,
                        use_resnet_dt=self._algo.use_resnet_dt)
                    H = tf.multiply(H, fc, name="H")
                else:
                    gtau = []
                    for tau in range(len(self._algo)):
                        with tf.name_scope(f"{tau}"):
                            v = self._algo.compute(
                                tau, rij, rc, dtype=dtype)
                            gx = tf.math.multiply(v, fc)
                            gtau.append(gx)
                    H = tf.concat(gtau, axis=-1, name="H")
                M = self.get_moment_tensor(
                    tf.expand_dims(rij, 0), dij, max_moment)
                T = self.get_T_dm(max_moment)
                P = tf.einsum("nback,dnbac->nbakd", H, M, name="P")
                S = tf.square(P, name="S")
                Q = tf.einsum("nbakd,dm->nabkm", S, T, name="Q")
                sign = tf.transpose(
                    tf.sign(P[:, :, :, :, 0]), [0, 2, 1, 3], name="s/sign")
                if max_moment == 0:
                    G = tf.multiply(
                        tf.sqrt(Q + eps), tf.expand_dims(sign, -1), name="G")
                else:
                    G = tf.concat([
                        tf.expand_dims(
                            tf.sqrt(Q[:, :, :, :, 0] + eps) * sign,
                            axis=4, name="m/0"),
                        Q[:, :, :, :, 1:]
                    ], axis=4, name="G")
                n = rij.shape.dims[0].value
                g = tf.reshape(G, (n, -1, ndims), name="rho")
                max_occurs[element] = g.shape.dims[1].value
                outputs[element] = g
        return outputs, max_occurs

    def calculate(self,
                  transformer: UniversalTransformer,
                  universal_descriptors,
                  mode: ModeKeys,
                  verbose=False) -> AtomicDescriptors:
        """
        Construct the computation graph for calculating descriptors.
        """
        with tf.name_scope("Radial"):
            if not self._legacy_mode:
                g, max_occurs = self.apply_model(
                    transformer, universal_descriptors["radial"], mode)
            else:
                partitions, max_occurs = dynamic_partition(
                    dists_and_masks=universal_descriptors['radial'],
                    elements=transformer.elements,
                    kbody_terms_for_element=transformer.kbody_terms_for_element,
                    mode=mode,
                    angular=False,
                    merge_symmetric=False)
                g = self.apply_legacy_pairwise_descriptor_functions(
                    transformer, partitions)
        return AtomicDescriptors(descriptors=g, max_occurs=max_occurs)
