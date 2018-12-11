# coding=utf-8
"""
This module defines various atomic neural networks.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import shutil
import json
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_io
from typing import List, Dict, Callable
from os.path import dirname, join

from tensoralloy.nn.atomic.normalizer import InputNormalizer
from tensoralloy.nn.utils import get_activation_fn, log_tensor
from tensoralloy.nn.basic import BasicNN
from tensoralloy.misc import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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
        self._normalizer = InputNormalizer(method=normalizer)

    @property
    def hidden_sizes(self) -> Dict[str, List[int]]:
        """
        Return the sizes of hidden layers for each element.
        """
        return self._hidden_sizes

    def _build_nn(self, features: AttributeDict, verbose=False):
        """
        Build 1x1 Convolution1D based atomic neural networks for all elements.

        Parameters
        ----------
        features : AttributeDict
            A dict of input tensors:
                * 'descriptors', a dict of (element, (value, mask)) where
                  `element` represents the symbol of an element, `value` is the
                  descriptors of `element` and `mask` is None.
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
        verbose : bool
            If True, the prediction tensors will be logged.

        """
        with tf.variable_scope("ANN"):
            activation_fn = get_activation_fn(self._activation)
            outputs = []
            for element, (value, _) in features.descriptors.items():
                with tf.variable_scope(element):
                    x = tf.identity(value, name='input')
                    if x.shape.ndims == 2:
                        x = tf.expand_dims(x, axis=0, name='2to3')
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
            A dict of input tensors.
        verbose : bool
            If True, the total energy tensor will be logged.

        Returns
        -------
        energy : tf.Tensor
            The total energy tensor.

        """
        with tf.name_scope("Energy"):
            y_atomic = tf.concat(outputs, axis=1, name='y_atomic')
            ndims = features.mask.shape.ndims
            axis = ndims - 1
            with tf.name_scope("mask"):
                if ndims == 1:
                    y_atomic = tf.squeeze(y_atomic, axis=0)
                mask = tf.split(
                    features.mask, [1, -1], axis=axis, name='split')[1]
                y_mask = tf.multiply(y_atomic, mask, name='mask')
            energy = tf.reduce_sum(
                y_mask, axis=axis, keepdims=False, name='energy')
            if verbose:
                log_tensor(energy)
            return energy

    def export(self, features_and_params_fn: Callable, output_graph_path: str,
               checkpoint=None, keep_tmp_files=True):
        """
        Freeze the graph and export the model to a pb file.

        Parameters
        ----------
        features_and_params_fn : Callable
            A `Callable` function to return (features, params).

            `features` should be a dict:
                * 'descriptors', a dict of (element, (value, mask)) where
                  `element` represents the symbol of an element, `value` is the
                  descriptors of `element` and `mask` is the mask of `value`.
                * 'positions' of shape `[batch_size, N, 3]`.
                * 'cells' of shape `[batch_size, 3, 3]`.
                * 'mask' of shape `[batch_size, N]`.
                * 'volume' of shape `[batch_size, ]`.
                * 'n_atoms' of dtype `int64`.'
            `params` should be a JSON dict returned by `Descriptor.as_dict`.

        output_graph_path : str
            The name of the output graph file.
        checkpoint : str or None
            The tensorflow checkpoint file to restore or None.
        keep_tmp_files : bool
            If False, the intermediate files will be deleted.

        """

        graph = tf.Graph()

        logdir = join(dirname(output_graph_path), 'export')
        input_graph_name = 'input_graph.pb'
        saved_model_ckpt = join(logdir, 'saved_model')
        saved_model_meta = f"{saved_model_ckpt}.meta"

        with graph.as_default():

            features, params = features_and_params_fn()
            predictions = self.build(features)

            # Encode the JSON dict into the graph.
            with tf.name_scope("Transformer/"):
                transformer_params = tf.constant(
                    json.dumps(params), name='params')

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver()
                if checkpoint is not None:
                    saver.restore(sess, checkpoint)
                checkpoint_path = saver.save(
                    sess, saved_model_ckpt, global_step=0)
                graph_io.write_graph(graph_or_graph_def=graph,
                                     logdir=logdir,
                                     name=input_graph_name)

            input_graph_path = join(logdir, input_graph_name)
            input_saver_def_path = ""
            input_binary = False
            restore_op_name = "save/restore_all"
            filename_tensor_name = "save/Const:0"
            clear_devices = True
            input_meta_graph = saved_model_meta

            output_node_names = [transformer_params.op.name]

            for tensor in predictions.values():
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
            shutil.rmtree(logdir)
