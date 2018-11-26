# coding=utf-8
"""
This module implements the nn-EAM.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

import tensorflow as tf


def _bytes_feature(value):
    """
    Convert the `value` to Protobuf bytes.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    Convert the `value` to Protobuf float32.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """
    Convert the `value` to Protobuf int64.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


