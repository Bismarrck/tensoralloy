# coding=utf-8
"""
The Tersoff potential.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from typing import List, Dict
from collections import Counter

from tensoralloy.utils import get_elements_from_kbody_term, get_kbody_terms
from tensoralloy.utils import GraphKeys, ModeKeys
from tensoralloy.nn.cutoff import tersoff_cutoff
from tensoralloy.io.lammps import read_tersoff_file
from tensoralloy.nn.dataclasses import EnergyOps, EnergyOp
from tensoralloy.nn.basic import BasicNN
from tensoralloy.nn.partition import dynamic_partition
from tensoralloy.nn.utils import log_tensor
from tensoralloy.transformer.universal import UniversalTransformer
from tensoralloy.precision import get_float_dtype
from tensoralloy.extension.grad_ops import safe_pow

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The default Tersoff parameters
default_tersoff_params = {
    "m": 3.0,
    "gamma": 1.0,
    "lambda3": 1.3258,
    "c": 4.8381,
    "d": 2.0417,
    "costheta0": 0.0,
    "n": 22.956,
    "beta": 0.33675,
    "lambda2": 1.3258,
    "B": 95.373,
    "R": 3.0,
    "D": 0.2,
    "lambda1": 3.2394,
    "A": 3264.7
}


class Tersoff(BasicNN):
    """
    The Tersoff interaction potential.
    """

    def __init__(self,
                 elements: List[str],
                 custom_potentials=None,
                 symmetric_mixing=False,
                 minimize_properties=('energy', 'forces'),
                 export_properties=('energy', 'forces', 'stress')):
        """
        Initialization method.
        """
        super(Tersoff, self).__init__(elements=elements, hidden_sizes=None,
                                      minimize_properties=minimize_properties,
                                      export_properties=export_properties)
        self._nn_scope = "Tersoff"
        self._custom_potentials = custom_potentials
        self._params = {}
        self._symmetric_mixing = symmetric_mixing
        self._variable_pool = {}
        self._read_initial_params()

    def as_dict(self):
        """
        Return a JSON serializable dict representation of this `BasicNN`.
        """
        return {"class": self.__class__.__name__,
                "elements": self._elements,
                "custom_potentials": self._custom_potentials,
                "symmetric_mixing": self._symmetric_mixing,
                "minimize_properties": self._minimize_properties,
                "export_properties": self._export_properties}

    def _read_initial_params(self):
        """
        Read the initial parameters.
        """
        if self._custom_potentials is None:
            for kbody_term in get_kbody_terms(
                    self._elements, angular=True, symmetric=False)[0]:
                if len(get_elements_from_kbody_term(kbody_term)) == 3:
                    self._params[kbody_term] = default_tersoff_params
        else:
            tersoff = read_tersoff_file(self._custom_potentials)
            assert self._elements == tersoff.elements
            for kbody_term, params in tersoff.params.items():
                self._params[kbody_term] = params

    def __get_powerintm(self, kbody_term):
        """
        Return the value of `m` for the corresponding kbody term. Currently, m
        can be only 1 or 3.
        """
        return int(self._params[kbody_term]['m'])

    def _get_or_create_variable(self, section, parameter, trainable=False):
        """
        Get or create the variable.
        """
        query = f"{section}.{parameter}"
        if query in self._variable_pool:
            return self._variable_pool[query]
        value = self._params[section][parameter]
        var = self._create_variable(parameter, value, trainable=trainable)
        self._variable_pool[query] = var
        return var

    def _get_radial_variable(self, kbody_term, parameter, trainable=True):
        """
        Get a radial variable.
        """
        element1, element2 = get_elements_from_kbody_term(kbody_term)
        if self._symmetric_mixing:
            if element1 < element2:
                full = f'{element1}{element2}{element2}'
            else:
                full = f'{element2}{element1}{element1}'
        else:
            full = f'{element1}{element2}{element2}'
        return self._get_or_create_variable(full, parameter, trainable)

    def _get_angular_variable(self, kbody_term, parameter, trainable=True):
        """
        Get an angular variable.
        """
        return self._get_or_create_variable(kbody_term, parameter, trainable)

    @staticmethod
    def _create_variable(parameter, value, trainable=True):
        """
        A helper function for creating a variable.
        """
        dtype = get_float_dtype()
        with tf.variable_scope(f"Parameters", reuse=tf.AUTO_REUSE):
            collections = [
                GraphKeys.TERSOFF_VARIABLES,
                tf.GraphKeys.MODEL_VARIABLES,
                tf.GraphKeys.GLOBAL_VARIABLES,
            ]
            if trainable:
                collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
            var = tf.get_variable(parameter,
                                  shape=(), dtype=dtype,
                                  initializer=tf.constant_initializer(
                                      value=value, dtype=dtype),
                                  regularizer=None,
                                  trainable=trainable,
                                  collections=collections,
                                  aggregation=tf.VariableAggregation.MEAN)
            return var

    def attach_transformer(self, clf: UniversalTransformer):
        """
        Attach an `UniversalTransformer` to this potential.

        Additional Tersoff checks:
            1. rcut >= max(R + D)
            2. rcut == acut

        """
        assert isinstance(clf, UniversalTransformer)
        rmax = 0.0
        dmax = 0.0
        for kbody_term, params in self._params.items():
            rmax = max(params['R'], rmax)
            dmax = max(params['D'], dmax)
        rdmax = rmax + dmax
        rcut = clf.rcut
        if rcut < rdmax:
            raise ValueError(f"The rcut {rcut:.1f} < max(R + D) = {rdmax:.1f}")
        if clf.rcut != clf.acut:
            raise ValueError(f"rcut {rcut:.1f} != acut {clf.acut:.1f}")

        super(Tersoff, self).attach_transformer(clf)

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
            return tf.concat(
                [tf.add_n(stacks[el], name=el) for el in self._elements],
                axis=1, name='sum')

    @staticmethod
    def _calculate_bij(bz, c1, c2, c3, c4, n, one, two):
        """
        A safe implementation for computing `bij` based on the Lammps
        implementation.
        """
        def func1(x):
            """ Segment 1: bz > c1 """
            return tf.math.divide_no_nan(one, tf.sqrt(x), name='y1')

        def func2(x):
            """ Segment 2: c1 >= bz > c2 """
            return tf.math.divide_no_nan(
                one - tf.pow(x, -n) / (two * n), tf.sqrt(x), name='y2')

        def func3(x):
            """ Segment 2: c2 >= bz > c3 """
            return tf.pow(one + tf.pow(x, n), -one / (two * n), name='y5')

        def func4(x):
            """ Segment 2: c3 >= bz > c4 """
            return tf.math.subtract(one, pow(x, n) / (two * n), name='y4')

        def func5(x):
            """ Segment 2: c4 >= bz """
            return tf.ones_like(x, dtype=x.dtype, name='y3')

        with tf.name_scope("bij"):
            idx1 = tf.where(tf.greater(bz, c1), name='idx1')
            idx2 = tf.where(tf.logical_and(
                tf.greater(bz, c2), tf.less_equal(bz, c1)), name='idx2')
            idx3 = tf.where(tf.logical_and(
                tf.greater(bz, c3), tf.less_equal(bz, c2)), name='idx3')
            idx4 = tf.where(tf.logical_and(
                tf.greater(bz, c4), tf.less_equal(bz, c3)), name='idx4')
            idx5 = tf.where(tf.less_equal(bz, c4), name='idx5')
            shape = tf.shape(bz, name='bz/shape', out_type=idx1.dtype)
            values = [
                tf.scatter_nd(idx1, func1(tf.gather_nd(bz, idx1)), shape),
                tf.scatter_nd(idx2, func2(tf.gather_nd(bz, idx2)), shape),
                tf.scatter_nd(idx3, func3(tf.gather_nd(bz, idx3)), shape),
                tf.scatter_nd(idx4, func4(tf.gather_nd(bz, idx4)), shape),
                tf.scatter_nd(idx5, func5(tf.gather_nd(bz, idx5)), shape),
            ]
            return tf.add_n(values, name='bij')

    def _get_model_outputs(self,
                           features: dict,
                           descriptors: dict,
                           mode: ModeKeys,
                           verbose=False):
        """
        Return raw NN-EAM model outputs.

        Parameters
        ----------
        features : Dict
            A dict of tensors, includeing raw properties and the descriptors:
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cell' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        descriptors : Dict
            A dict of (element, (dists, masks)) where `element` represents the
            symbol of an element.
        mode : ModeKeys
            Specifies if this is training, evaluation or prediction.
        verbose : bool
            If True, the prediction tensors will be logged.

        Returns
        -------
        y : tf.Tensor
            A 1D (PREDICT) or 2D (TRAIN or EVAL) tensor as the unmasked atomic
            energies of atoms. The last axis has the size `max_n_atoms`.

        """
        with tf.variable_scope(self._nn_scope, reuse=tf.AUTO_REUSE):
            clf = self._transformer
            dtype = get_float_dtype()
            assert isinstance(clf, UniversalTransformer)

            with tf.name_scope("Constants"):
                two = tf.convert_to_tensor(2.0, dtype=dtype, name='two')
                one = tf.convert_to_tensor(1.0, dtype=dtype, name='one')
                half = tf.convert_to_tensor(0.5, dtype=dtype, name='half')

            zeta_dict: Dict = {}

            with tf.name_scope("Angular"):
                angular = dynamic_partition(
                    dists_and_masks=descriptors["angular"],
                    elements=clf.elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    merge_symmetric=False,
                    angular=True)[0]

                for kbody_term, (dists, masks) in angular.items():
                    center, mid = get_elements_from_kbody_term(kbody_term)[0: 2]
                    with tf.variable_scope(f"{kbody_term}"):
                        gamma = self._get_angular_variable(kbody_term, "gamma")
                        c = self._get_angular_variable(kbody_term, "c")
                        d = self._get_angular_variable(kbody_term, "d")
                        costheta0 = self._get_angular_variable(
                            kbody_term, "costheta0")
                        lambda3 = self._get_angular_variable(
                            kbody_term, "lambda3")
                        with tf.variable_scope("Fixed"):
                            R = self._get_angular_variable(
                                kbody_term, "R", trainable=False)
                            D = self._get_angular_variable(
                                kbody_term, "D", trainable=False)
                            m = self._get_angular_variable(
                                kbody_term, "m", trainable=False)
                            powerintm = self.__get_powerintm(kbody_term)
                        c2 = tf.square(c, name='c2')
                        d2 = tf.square(d, name='d2')
                        masks = tf.squeeze(masks, axis=1, name='masks')
                        rij = tf.squeeze(dists[0], axis=1, name='rij')
                        rik = tf.squeeze(dists[4], axis=1, name='rik')
                        rjk = tf.squeeze(dists[8], axis=1, name='rjk')
                        rij2 = tf.square(rij, name='rij2')
                        rik2 = tf.square(rik, name='rik2')
                        rjk2 = tf.square(rjk, name='rjk2')
                        upper = tf.math.subtract(
                            rij2 + rik2, rjk2, name='upper')
                        lower = tf.math.multiply(
                            tf.math.multiply(two, rij), rik, name='lower')
                        costheta = tf.math.divide_no_nan(
                            upper, lower, name='costheta')
                        right = c2 / d2 - c2 / (d2 + (costheta - costheta0)**2)
                        gtheta = tf.math.multiply(
                            gamma, tf.math.add(one, right),
                            name='gtheta')
                        fc = tersoff_cutoff(rik, R, D, name='fc')
                        drijk = tf.math.subtract(rij, rik)
                        with tf.name_scope("Safe"):
                            arg = tf.math.multiply(drijk, lambda3)
                            ub = tf.constant(69.0776 / float(powerintm),
                                             name='ub', dtype=R.dtype)
                            lb = tf.negative(ub, name='lb')
                            arg = tf.clip_by_value(arg, lb, ub, name='arg')
                            if powerintm == 3:
                                arg = safe_pow(arg, m)
                            arg = tf.math.exp(arg)
                        z = tf.math.multiply(fc * gtheta,
                                             arg,
                                             name='zeta/single')
                        z = tf.math.multiply(z, masks, name='zeta/masked')
                        z = tf.reduce_sum(
                            z, axis=-1, keepdims=True, name='zeta')
                        key = f'{center}{mid}'
                        zeta_dict[key] = zeta_dict.get(key, []) + [z]

            outputs = {}

            with tf.name_scope("Radial"):
                radial, max_occurs = dynamic_partition(
                    dists_and_masks=descriptors["radial"],
                    elements=self._elements,
                    kbody_terms_for_element=clf.kbody_terms_for_element,
                    mode=mode,
                    angular=False,
                    merge_symmetric=False)
                for kbody_term, (dists, masks) in radial.items():
                    center, pair = get_elements_from_kbody_term(kbody_term)
                    with tf.variable_scope(f"{kbody_term}"):
                        with tf.variable_scope("Fixed"):
                            R = self._get_radial_variable(
                                kbody_term, "R", trainable=False)
                            D = self._get_radial_variable(
                                kbody_term, "D", trainable=False)
                        A = self._get_radial_variable(kbody_term, "A")
                        B = self._get_radial_variable(kbody_term, "B")
                        lambda1 = self._get_radial_variable(
                            kbody_term, "lambda1")
                        lambda2 = self._get_radial_variable(
                            kbody_term, "lambda2")
                        beta = self._get_radial_variable(kbody_term, "beta")
                        n = self._get_radial_variable(kbody_term, "n")
                        with tf.name_scope("Safe"):
                            c0 = tf.constant(1e-8, dtype=R.dtype, name='c0')
                            c1 = tf.pow(c0 * c0 * two * n, -one / n, name='c1')
                            c2 = tf.pow(c0 * two * n, -one / n, name='c2')
                            c3 = tf.math.divide_no_nan(one, c2, name='c3')
                            c4 = tf.math.divide_no_nan(one, c1, name='c4')

                        masks = tf.squeeze(masks, axis=1, name='masks')
                        rij = tf.squeeze(dists[0], axis=1, name='rij')
                        fc = tersoff_cutoff(rij, R, D, name='fc')
                        fr = tf.math.multiply(
                            A, tf.exp(-lambda1 * rij), name='fr')
                        repulsion = tf.math.multiply(fc, fr, name='repulsion')
                        fa = tf.math.multiply(
                            -B, tf.exp(-lambda2 * rij), name='fa')
                        zeta = tf.add_n(
                            zeta_dict[f'{center}{pair}'], name='zeta')
                        bz = tf.math.multiply(beta, zeta, name='bz')
                        bij = self._calculate_bij(
                            bz, c1, c2, c3, c4, n, one, two)
                        attraction = tf.math.multiply(fc, bij * fa,
                                                      name='attraction')
                        vij = tf.math.add(repulsion, attraction, name='vij')
                        vij = tf.math.multiply(half, vij, name='vij/half')
                        vij = tf.math.multiply(vij, masks, name='vij/masked')
                        outputs[kbody_term] = tf.reduce_sum(
                            vij, axis=[-1, -2], keepdims=False, name='vij/sum')

            y_atomic = self._dynamic_stitch(outputs, max_occurs)
            if mode == ModeKeys.PREDICT or mode == ModeKeys.LAMMPS:
                y_atomic = tf.squeeze(y_atomic, name='atomic/squeeze')
            return y_atomic

    def _get_energy_ops(self, outputs: tf.Tensor, features: dict, verbose=True):
        """
        Return the Op to compute internal energy E.

        Parameters
        ----------
        outputs : tf.Tensor
            A 2D tensor of shape `[batch_size, max_n_atoms - 1]` as the unmasked
            atomic energies.
        features : Dict
            A dict of input features.
        name : str
            The name of the output tensor.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        ops : tf.Tensor
            The energy tensors.

        """
        eatom = tf.identity(outputs, name='atomic/raw')
        ndims = features["atom_masks"].shape.ndims
        axis = ndims - 1
        with tf.name_scope("Mask"):
            mask = tf.split(
                features["atom_masks"], [1, -1], axis=axis, name='split')[1]
        eatom = tf.multiply(eatom, mask, name='atomic')
        energy = tf.reduce_sum(
            eatom, axis=axis, keepdims=False, name='energy')
        if verbose:
            log_tensor(energy)
        return EnergyOps(EnergyOp(total=energy, atomic=eatom))
