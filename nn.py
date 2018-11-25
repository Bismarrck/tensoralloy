# coding=utf-8
"""
This module defines a general atomic neural network framework.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.opt import NadamOptimizer
from typing import List, Dict

from misc import Defaults, AttributeDict
from tensoralloy.nn.basic import BasicNN

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class GraphKeys:
    """
    Standard names for variable collections.
    """

    # Summary keys
    TRAIN_SUMMARY = 'train_summary'
    EVAL_SUMMARY = 'eval_summary'

    # Variable keys
    NORMALIZE_VARIABLES = 'normalize_vars'
    STATIC_ENERGY_VARIABLES = 'static_energy_vars'

    # Metrics Keys
    TRAIN_METRICS = 'train_metrics'
    EVAL_METRICS = 'eval_metrics'

    # Gradient Keys
    ENERGY_GRADIENTS = 'energy_gradients'
    FORCES_GRADIENTS = 'forces_gradients'
    STRESS_GRADIENTS = 'stress_gradients'


def get_activation_fn(fn_name: str):
    """
    Return the corresponding activation function.
    """
    if fn_name.lower() == 'leaky_relu':
        return tf.nn.leaky_relu
    elif fn_name.lower() == 'relu':
        return tf.nn.relu
    elif fn_name.lower() == 'tanh':
        return tf.nn.tanh
    elif fn_name.lower() == 'sigmoid':
        return tf.nn.sigmoid
    elif fn_name.lower() == 'softplus':
        return tf.nn.softplus
    elif fn_name.lower() == 'softsign':
        return tf.nn.softsign
    elif fn_name.lower() == 'softmax':
        return tf.nn.softmax
    raise ValueError("The function '{}' cannot be recognized!".format(fn_name))


def get_learning_rate(global_step, learning_rate=0.001, decay_function=None,
                      decay_rate=0.99, decay_steps=1000, staircase=False):
    """
    Return a `float64` tensor as the learning rate.
    """
    with tf.name_scope("learning_rate"):
        if decay_function is None:
            learning_rate = tf.constant(
                learning_rate, dtype=tf.float64, name="learning_rate")
        else:
            if decay_function == 'exponential':
                decay_fn = tf.train.exponential_decay
            elif decay_function == 'inverse_time':
                decay_fn = tf.train.inverse_time_decay
            elif decay_function == 'natural_exp':
                decay_fn = tf.train.natural_exp_decay
            else:
                raise ValueError(
                    "'{}' is not supported!".format(decay_function))
            learning_rate = decay_fn(learning_rate,
                                     global_step=global_step,
                                     decay_rate=decay_rate,
                                     decay_steps=decay_steps,
                                     staircase=staircase,
                                     name="learning_rate")
        tf.summary.scalar('learning_rate_at_step', learning_rate,
                          collections=[GraphKeys.TRAIN_SUMMARY, ])
        return learning_rate


def get_optimizer(learning_rate, method='adam', **kwargs):
    """
    Return a `tf.train.Optimizer`.

    Parameters
    ----------
    learning_rate : tf.Tensor or float
        A float tensor as the learning rate.
    method : str
        A str as the name of the optimizer. Supported are: 'adam', 'adadelta',
        'rmsprop' and 'nadam'.
    kwargs : dict
        Additional arguments for the optimizer.

    Returns
    -------
    optimizer : tf.train.Optimizer
        An optimizer.

    """
    with tf.name_scope("SGD"):
        if method.lower() == 'adam':
            return tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=kwargs.get('beta1', 0.9))
        elif method.lower() == 'nadam':
            return NadamOptimizer(
                learning_rate=learning_rate, beta1=kwargs.get('beta1', 0.9))
        elif method.lower() == 'adadelta':
            return tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate, rho=kwargs.get('rho', 0.95))
        elif method.lower() == 'rmsprop':
            return tf.train.RMSPropOptimizer(
                learning_rate=learning_rate, decay=kwargs.get('decay', 0.9),
                momentum=kwargs.get('momentum', 0.0))
        else:
            raise ValueError(
                "Supported SGD optimizers: adam, nadam, adadelta, rmsprop.")


def log_tensor(tensor: tf.Tensor):
    """
    Print the name and shape of the input Tensor.
    """
    dimensions = ",".join(["{:6d}".format(dim if dim is not None else -1)
                           for dim in tensor.get_shape().as_list()])
    tf.logging.info("{:<40s} : [{}]".format(tensor.op.name, dimensions))


def msra_initializer(dtype=tf.float64, seed=Defaults.seed):
    """
    Return the so-called `MSRA` initializer.

    See Also
    --------
    [Delving Deep into Rectifiers](http://arxiv.org/pdf/1502.01852v1.pdf)

    """
    return variance_scaling_initializer(dtype=dtype, seed=seed)


class _InputNormalizer:
    """
    A collection of funcitons for normalizing input descriptors to [0, 1].
    """

    def __init__(self, method='linear'):
        """
        Initialization method.
        """
        assert method in ('linear', 'arctan', '', None)
        if not method:
            self.method = None
            self.scope = 'Identity'
        else:
            self.method = method
            self.scope = '{}Norm'.format(method.capitalize())

    def __call__(self, x: tf.Tensor, values=None):
        """
        Apply the normalization.
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if not self.method:
                return tf.identity(x, name='identity')

            if values is None:
                values = np.ones(x.shape[2], dtype=np.float64)
            alpha = tf.get_variable(
                name='alpha',
                shape=x.shape[2],
                dtype=x.dtype,
                initializer=tf.constant_initializer(values, dtype=x.dtype),
                collections=[tf.GraphKeys.TRAINABLE_VARIABLES,
                             tf.GraphKeys.GLOBAL_VARIABLES,
                             GraphKeys.NORMALIZE_VARIABLES],
                trainable=True)
            tf.summary.histogram(
                name=alpha.op.name + '/summary',
                values=alpha,
                collections=[GraphKeys.TRAIN_SUMMARY])
            x = tf.multiply(x, alpha, name='ax')
            if self.method == 'linear':
                x = tf.identity(x, name='x')
            elif self.method == 'arctan':
                x = tf.atan(x, name='x')
            else:
                raise ValueError(
                    f"Unsupported normalization method: {self.method}")
            tf.summary.histogram(
                name=x.op.name + '/summary',
                values=x,
                collections=[GraphKeys.TRAIN_SUMMARY])
            return x


class AtomicNN(BasicNN):
    """
    This class represents a general atomic neural network.
    """

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 forces=False, stress=False, total_pressure=False, l2_weight=0.,
                 normalizer=None, normalization_weights=None):
        """
        Initialization method.

        normalizer : str
            The normalization method. Defaults to 'linear'. Set this to None to
            disable normalization.
        normalization_weights : Dict[str, array_like]
            The initial weights for column-wise normalizing the input atomic
            descriptors.

        """
        super(AtomicNN, self).__init__(
            elements=elements, hidden_sizes=hidden_sizes, activation=activation,
            forces=forces, stress=stress, total_pressure=total_pressure,
            l2_weight=l2_weight)
        self._initial_normalizer_weights = normalization_weights
        self._normalizer = _InputNormalizer(method=normalizer)

    @property
    def hidden_sizes(self) -> Dict[str, List[int]]:
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Build 1x1 Convolution1D based atomic neural networks for all elements.
        """
        with tf.variable_scope("ANN"):
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for i, element in enumerate(self._elements):
                with tf.variable_scope(element):
                    x = tf.identity(features.descriptors[element], name='input')
                    if self._initial_normalizer_weights is not None:
                        x = self._normalizer(
                            x, self._initial_normalizer_weights[element])
                    hidden_sizes = self._hidden_sizes[element]
                    if verbose:
                        log_tensor(x)
                    yi = self._get_1x1conv_nn(
                        x, activation_fn, hidden_sizes, verbose=verbose)
                    yi = tf.squeeze(yi, axis=2, name='atomic')
                    if verbose:
                        log_tensor(yi)
                    outputs.append(yi)
            return outputs

    def _get_energy(self, outputs, features, verbose=True):
        """
        Return the Op to compute total energy.

        Parameters
        ----------
        outputs : List[tf.Tensor]
            A list of `tf.Tensor` as the outputs of the ANNs.
        features : AttributeDict
            A dict of input features.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        with tf.name_scope("Energy"):
            y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
            shape = y_atomic.shape
            with tf.name_scope("mask"):
                mask = tf.split(
                    features.mask, [1, -1], axis=1, name='split')[1]
                y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_mask.set_shape(shape)
            energy = tf.reduce_sum(
                y_mask, axis=1, keepdims=False, name='energy')
            if verbose:
                log_tensor(energy)
            return energy


class AtomicResNN(AtomicNN):
    """
    A general atomic residual neural network. The ANNs are used to fit residual
    energies.

    The total energy, `y_total`, is expressed as:

        y_total = y_static + y_res

    where `y_static` is the total atomic static energy and this only depends on
    chemical compositions:

        y_static = sum([atomic_static_energy[e] * count(e) for e in elements])

    where `count(e)` represents the number of element `e`.

    """

    def __init__(self, elements: List[str], hidden_sizes=None, activation=None,
                 l2_weight=0., forces=False, stress=False, total_pressure=False,
                 normalizer='linear', normalization_weights=None,
                 atomic_static_energy=None):
        """
        Initialization method.
        """
        super(AtomicResNN, self).__init__(
            elements=elements, hidden_sizes=hidden_sizes, activation=activation,
            l2_weight=l2_weight, forces=forces, stress=stress,
            total_pressure=total_pressure, normalizer=normalizer,
            normalization_weights=normalization_weights)
        self._atomic_static_energy = atomic_static_energy

    def _check_keys(self, features: AttributeDict, labels: AttributeDict):
        """
        Check the keys of `features` and `labels`.
        """
        super(AtomicResNN, self)._check_keys(features, labels)
        assert 'composition' in features

    def _get_energy(self, outputs, features, verbose=True):
        """
        Return the Op to compute total energy (eV).
        """
        with tf.name_scope("Energy"):

            with tf.variable_scope("Static", reuse=tf.AUTO_REUSE):
                if self._atomic_static_energy is None:
                    values = np.ones(len(self._elements), dtype=np.float64)
                else:
                    values = np.asarray(
                        [self._atomic_static_energy[e] for e in self._elements],
                        dtype=np.float64)
                initializer = tf.constant_initializer(values, dtype=tf.float64)
                x = tf.identity(features.composition, name='input')
                z = tf.get_variable("weights",
                                    shape=(len(self._elements)),
                                    dtype=tf.float64,
                                    trainable=True,
                                    collections=[
                                        GraphKeys.STATIC_ENERGY_VARIABLES,
                                        GraphKeys.TRAIN_METRICS,
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        tf.GraphKeys.GLOBAL_VARIABLES],
                                    initializer=initializer)
                xz = tf.multiply(x, z, name='xz')
                y_static = tf.reduce_sum(xz, axis=1, keepdims=False,
                                         name='static')
                if verbose:
                    log_tensor(y_static)

            with tf.name_scope("Residual"):
                y_atomic = tf.concat(outputs, axis=1, name='atomic')
                with tf.name_scope("mask"):
                    mask = tf.split(
                        features.mask, [1, -1], axis=1, name='split')[1]
                    y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_res = tf.reduce_sum(y_mask, axis=1, keepdims=False,
                                      name='residual')

            energy = tf.add(y_static, y_res, name='energy')

            with tf.name_scope("Ratio"):
                ratio = tf.reduce_mean(tf.div(y_static, energy, name='ratio'),
                                       name='avg')
                tf.add_to_collection(GraphKeys.TRAIN_METRICS, ratio)
                tf.summary.scalar(ratio.op.name + '/summary', ratio,
                                  collections=[GraphKeys.TRAIN_SUMMARY, ])

            return energy


class EamNN(BasicNN):
    """
    The implementation of nn-EAM.
    """

    def _get_energy(self, outputs: (List[tf.Tensor], List[tf.Tensor]),
                    features: AttributeDict, verbose=True):
        """
        Return the Op to compute total energy of nn-EAM.

        Parameters
        ----------
        outputs : List[List[tf.Tensor], List[tf.Tensor]]
            A tuple of `List[tf.Tensor]`.
        features : AttributeDict
            A dict of input features.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        with tf.name_scope("Energy"):
            with tf.name_scope("Atomic"):
                values = []
                for i, (phi, embed) in enumerate(zip(*outputs)):
                    values.append(tf.add(phi, embed, name=self._elements[i]))
                y_atomic = tf.concat(values, axis=1, name='atomic')
            shape = y_atomic.shape
            with tf.name_scope("mask"):
                mask = tf.split(
                    features.mask, [1, -1], axis=1, name='split')[1]
                y_mask = tf.multiply(y_atomic, mask, name='mask')
                y_mask.set_shape(shape)
            energy = tf.reduce_sum(
                y_mask, axis=1, keepdims=False, name='energy')
            if verbose:
                log_tensor(energy)
            return energy

    def _build_phi_nn(self, features: AttributeDict, verbose=False):
        """
        Return the outputs of the pairwise interactions, `Phi(r)`.
        """
        activation_fn = get_activation_fn(self._activation)
        outputs = []
        with tf.name_scope("Phi"):
            for i, element in enumerate(self._elements):
                with tf.variable_scope(element):
                    # Convert `x` to a 5D tensor.
                    x = tf.expand_dims(features.descriptors[element],
                                       axis=-1, name='input')
                    if verbose:
                        log_tensor(x)
                    hidden_sizes = self._hidden_sizes[element]
                    y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                             verbose=verbose)
                    # `y` here will be reduced to a 2D tensor of shape
                    # `[batch_size, max_n_atoms]`
                    y = tf.reduce_sum(y, axis=(2, 3, 4), keepdims=False,
                                      name='atomic')
                    if verbose:
                        log_tensor(y)
                    outputs.append(y)
            return outputs

    def _build_embed_nn(self, features: AttributeDict, verbose=False):
        """
        Return the outputs of the embedding energy, `F(rho(r))`.
        """
        activation_fn = get_activation_fn(self._activation)
        outputs = []
        with tf.name_scope("Embed"):
            for i, element in enumerate(self._elements):
                hidden_sizes = self._hidden_sizes[element]
                with tf.variable_scope(element):
                    with tf.variable_scope("Rho"):
                        x = tf.expand_dims(features.descriptors[element],
                                           axis=-1, name='input')
                        if verbose:
                            log_tensor(x)
                        y = self._get_1x1conv_nn(x, activation_fn, hidden_sizes,
                                                 verbose=verbose)
                        rho = tf.reduce_sum(y, axis=(3, 4), keepdims=False)
                        if verbose:
                            log_tensor(rho)
                    with tf.variable_scope("F"):
                        rho = tf.expand_dims(rho, axis=-1, name='rho')
                        y = self._get_1x1conv_nn(
                            rho, activation_fn, hidden_sizes, verbose=verbose)
                        embed = tf.reduce_sum(y, axis=(2, 3), name='embed')
                        if verbose:
                            log_tensor(embed)
                        outputs.append(embed)
            return outputs

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Return the nn-EAM model.
        """
        with tf.name_scope("nnEAM"):
            outputs = (
                self._build_phi_nn(features, verbose=verbose),
                self._build_embed_nn(features, verbose=verbose)
            )
            return outputs
