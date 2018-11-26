# coding=utf-8
"""
This module defines interfaces for feature transformers.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import abc
from collections import Counter
from typing import Dict
from ase import Atoms

from tensoralloy.misc import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class DescriptorTransformer:
    """
    This interface class defines the required methods for atomic descriptor
    transformers.
    """

    @property
    @abc.abstractmethod
    def placeholders(self) -> Dict[str, tf.Tensor]:
        """
        Return a dict of names and placeholders.
        """
        pass

    @abc.abstractmethod
    def get_feed_dict(self, atoms: Atoms):
        """
        Return a feed dict.
        """
        pass

    @abc.abstractmethod
    def get_graph(self):
        """
        Return the graph for computing atomic descriptors.
        """
        pass


class BatchDescriptorTransformer:
    """
    This interface class defines the required methods for atomic descriptor
    transformers.
    """

    @property
    @abc.abstractmethod
    def forces(self):
        """
        Return True if atomic forces should be encoded and trained.
        """
        pass

    @property
    @abc.abstractmethod
    def stress(self):
        """
        Return True if the stress tensor should be encoded and trained.
        """
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        """
        Return the batch size.
        """
        pass

    @property
    @abc.abstractmethod
    def max_occurs(self) -> Counter:
        """
        Return the maximum occurance of each type of element.
        """
        pass

    @abc.abstractmethod
    def encode(self, atoms: Atoms) -> tf.train.Example:
        """
        Encode the `Atoms` object to a tensorflow example.
        """
        pass

    @abc.abstractmethod
    def decode_protobuf(self, example_proto: tf.Tensor) -> AttributeDict:
        """
        Decode the scalar string Tensor, which is a single serialized Example.
        See `_parse_single_example_raw` documentation for more details.
        """
        pass

    @abc.abstractmethod
    def get_graph_from_batch(self, batch: AttributeDict, batch_size: int):
        """
        Return the tensorflow graph for computing atomic descriptors from an
        input batch.
        """
        pass

    @abc.abstractmethod
    def get_descriptor_normalization_weights(self, method):
        """
        Return the initial weights for column-wise normalising the output atomic
        descriptors.
        """
        pass
