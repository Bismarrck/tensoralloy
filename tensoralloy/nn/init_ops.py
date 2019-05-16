#!coding=utf-8
"""
This module defines weights initializers.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from tensorflow.python.keras.initializers import VarianceScaling, RandomNormal
from tensorflow.python.keras.initializers import RandomUniform, TruncatedNormal
from tensorflow.python.keras.initializers import GlorotNormal, GlorotUniform


from tensoralloy.utils import Defaults

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def he_normal(seed=None, dtype=tf.float32):
  """
  Kaiming He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.
  """
  return VarianceScaling(
      scale=2., mode="fan_in", distribution="truncated_normal", seed=seed,
      dtype=dtype)


def he_uniform(seed=None, dtype=tf.float32):
  """
  Kaiming He uniform variance scaling initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.
  """
  return VarianceScaling(
      scale=2., mode="fan_in", distribution="uniform", seed=seed, dtype=dtype)


def lecun_normal(seed=None, dtype=tf.float32):
  """LeCun normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(1 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.
  """
  return VarianceScaling(
      scale=1., mode="fan_in", distribution="truncated_normal", seed=seed,
      dtype=dtype)


def lecun_uniform(seed=None, dtype=tf.float32):
  """LeCun uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(3 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.
  """
  return VarianceScaling(
      scale=1., mode="fan_in", distribution="uniform", seed=seed, dtype=dtype)


random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
glorot_uniform_initializer = GlorotUniform
xavier_uniform_initializer = GlorotUniform
glorot_normal_initializer = GlorotNormal
xavier_normal_initializer = GlorotUniform
he_normal_initializer = he_normal
he_uniform_initializer = he_uniform
lecun_uniform_initializer = lecun_uniform
lecun_normal_initializer = lecun_normal


def get_initializer(name: str, dtype=tf.float64, seed=Defaults.seed):
    """
    Return a variable initializer.
    """
    name = name.lower()
    if name == 'random_uniform':
        init_fn = random_uniform_initializer
    elif name == 'random_normal':
        init_fn = random_normal_initializer
    elif name == 'truncated_normal':
        init_fn = truncated_normal_initializer
    elif name == 'glorot_uniform' or name == 'xavier_uniform':
        init_fn = glorot_uniform_initializer
    elif name == 'glorot_normal' or name == 'xavier_normal':
        init_fn = glorot_normal_initializer
    elif name == 'he_normal':
        init_fn = he_normal_initializer
    elif name == 'he_uniform':
        init_fn = he_uniform_initializer
    elif name == 'lecun_uniform':
        init_fn = lecun_uniform_initializer
    elif name == 'lecun_normal':
        init_fn = lecun_normal_initializer
    elif name == 'zero':
        return tf.zeros_initializer(dtype=dtype)
    else:
        raise ValueError(f"The initializer {name} cannot be recognized!")
    return init_fn(dtype=dtype, seed=seed)
