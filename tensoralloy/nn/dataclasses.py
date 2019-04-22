#!coding=utf-8
"""
This module defines data classes for `tensoralloy.nn` package.
"""
from __future__ import print_function, absolute_import

from dataclasses import dataclass
from typing import List, Union

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass
class _LossOptions:
    """
    The basic options for a loss.
    """
    weight: float = 1.0


@dataclass
class EnergyLossOptions(_LossOptions):
    """
    Special options for the loss of energies.
    """
    per_atom_loss: bool = False


@dataclass
class ForcesLossOptions(_LossOptions):
    pass


@dataclass
class StressLossOptions(_LossOptions):
    """
    Special options for the loss of stress tensors.
    """
    use_rmse: bool = True


@dataclass
class PressureLossOptions(_LossOptions):
    pass


@dataclass
class L2LossOptions(_LossOptions):
    """
    Special options for the L2 regularization.
    The default weight is changed to 0.01.
    """
    weight: float = 0.01


@dataclass
class ElasticConstraintOptions:
    """
    Options for computing loss of the elastic contraints.
    """
    use_kbar: bool = True
    forces_weight: float = 1.0
    stress_weight: float = 0.1


@dataclass
class ElasticLossOptions(_LossOptions):
    """
    Special options for the loss of elastic constants.
    """
    crystals: List[str] = None
    constrain_options: ElasticConstraintOptions = ElasticConstraintOptions()


@dataclass
class LossParameters:
    """
    Hyper parameters for constructing the total loss.
    """

    equivalently_trusted: bool = True
    energy: EnergyLossOptions = EnergyLossOptions()
    forces: ForcesLossOptions = ForcesLossOptions()
    stress: StressLossOptions = StressLossOptions()
    total_pressure: PressureLossOptions = PressureLossOptions()
    l2: L2LossOptions = L2LossOptions()
    elastic: ElasticLossOptions = ElasticLossOptions()


@dataclass
class OptParameters:
    """
    Hyper parameters for optimizing the total loss.
    """
    method: str = 'adam'
    learning_rate: float = 0.01
    decay_function: Union[str, None] = None
    decay_rate: float = 0.99
    decay_steps: int = 1000
    staircase: bool = False


@dataclass(init=True, frozen=True)
class StructuralProperty:
    """
    A type of property of a structure.

    Attributes
    ----------
    name : str
        The name of this property.
    minimizable : bool
        A boolean indicating whether this property can be minimized or not.
    exportable : bool
        A boolean indicating whether this property can be exported or not.

    """

    name: str
    minimizable: bool = True
    exportable: bool = True

    def __eq__(self, other):
        if hasattr(other, "name"):
            return other.name == self.name
        else:
            return str(other) == self.name
