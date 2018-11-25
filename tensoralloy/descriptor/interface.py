# coding=utf-8
"""
This module defines the abstract class of all atomic descriptors.
"""
from __future__ import print_function, absolute_import

import abc
from typing import List

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
    def build_graph(self, placeholders: AttributeDict):
        """
        Build the tensorflow graph for computing atomic descriptors.
        """
        pass
