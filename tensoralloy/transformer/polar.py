#!coding=utf-8
"""
The special transformer for predicting the polar tensor.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from typing import Dict

from tensoralloy.transformer.base import bytes_feature
from tensoralloy.transformer import BatchUniversalTransformer
from tensoralloy.atoms_utils import get_polar_tensor
from tensoralloy.precision import get_float_dtype

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class BatchPolarTransformer(BatchUniversalTransformer):

    def _encode_additional_properties(self, atoms):
        np_dtype = get_float_dtype().as_numpy_dtype
        polar = get_polar_tensor(atoms).astype(np_dtype)
        return {"polar": bytes_feature(polar.tostring())}

    @staticmethod
    def _decode_additional_properties(example: Dict[str, tf.Tensor]):
        dtype = get_float_dtype()
        polar = tf.decode_raw(example["polar"], dtype)
        polar.set_shape([6])
        polar = tf.reshape(polar, (6, ), name='polar')
        return {"polar": polar}

    def get_decode_feature_list(self):
        feature_list = super(
            BatchPolarTransformer, self).get_decode_feature_list()
        feature_list['polar'] = tf.FixedLenFeature([], tf.string)
        return feature_list
