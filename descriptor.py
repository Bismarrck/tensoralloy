# coding=utf-8
"""
This module defines the base class for all atomic descriptors.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import abc
from ase import Atoms
from typing import List, Dict
from misc import AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AtomicDescriptor(abc.ABC):
    """
    The base interface class for all kinds of atomic descriptors.
    """

    @property
    @abc.abstractmethod
    def cutoff(self) -> float:
        """
        Return the cutoff radius.
        """
        pass

    @property
    @abc.abstractmethod
    def elements(self) -> List[str]:
        """
        Return a list of str as the ordered unique elements.
        """
        pass

    @abc.abstractmethod
    def get_graph(self, placeholders: AttributeDict):
        """
        Return the tensorflow graph for computing atomic descriptors.
        """
        pass


class DescriptorTransformer(AtomicDescriptor):
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


class BatchDescriptorTransformer(AtomicDescriptor):
    """
    This interface class defines the required methods for atomic descriptor
    transformers.
    """

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        """
        Return the batch size.
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
    def get_graph_from_batch(self, batch: AttributeDict):
        """
        Return the tensorflow graph for computing atomic descriptors from an
        input batch.
        """
        pass
