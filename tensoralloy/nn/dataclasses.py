#!coding=utf-8
"""
This module defines data classes for `tensoralloy.nn` package.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from dataclasses import dataclass, is_dataclass
from typing import List, Union, Dict, Tuple

from tensoralloy.utils import add_slots

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def nested_dataclass(*args, **kwargs):
    """
    A decorator to create a nested dataclass.

    References
    ----------
    https://stackoverflow.com/questions/51564841

    """
    def wrapper(cls):
        cls = dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                     new_obj = field_type(**value)
                     kwargs[name] = new_obj
            original_init(self, *args, **kwargs)
        cls.__init__ = __init__
        return cls
    return wrapper(args[0]) if args else wrapper


@dataclass(frozen=True)
class EnergyOps:
    """
    Different types of energy ops.

    energy: the internal energy U
    eentropy: the electron entropy S
    enthalpy: the enthalpy H = U + PV
    free_energy: electron free energy F = U - T*S
    atomic: energy for each atom

    """
    energy: tf.Tensor
    eentropy: tf.Tensor
    enthalpy: tf.Tensor
    free_energy: tf.Tensor
    atomic: tf.Tensor


@add_slots
@dataclass
class _LossOptions:
    """
    The basic options for a loss.
    """
    weight: Union[float, Tuple[float, float]] = 1.0


@add_slots
@dataclass
class EnergyLossOptions(_LossOptions):
    """
    Special options for the loss of energies.
    """
    per_atom_loss: bool = False
    method: str = 'rmse'


@add_slots
@dataclass
class ForcesLossOptions(_LossOptions):
    """
    Special options for the loss of atomic forces.
    """
    method: str = 'rmse'


@add_slots
@dataclass
class StressLossOptions(_LossOptions):
    """
    Special options for the loss of stress tensors.
    """
    method: str = 'rmse'


@add_slots
@dataclass
class PressureLossOptions(_LossOptions):
    """
    Special options for the loss of total pressures.
    """
    method: str = 'rmse'


@add_slots
@dataclass
class L2LossOptions(_LossOptions):
    """
    Special options for the L2 regularization.
    The default weight is changed to 0.01.
    """
    weight: float = 0.01
    decayed: bool = True
    decay_rate: float = 0.99
    decay_steps: int = 1000


@add_slots
@dataclass
class ElasticConstraintOptions:
    """
    Options for computing loss of the elastic contraints.
    """
    use_kbar: bool = True
    forces_weight: float = 1.0
    stress_weight: float = 0.1
    tau: float = 1.0


@add_slots
@nested_dataclass
@dataclass
class ElasticLossOptions(_LossOptions):
    """
    Special options for the loss of elastic constants.
    """
    crystals: List[str] = None
    constraint: ElasticConstraintOptions = ElasticConstraintOptions()


@add_slots
@dataclass
class RoseLossOptions(_LossOptions):
    """
    Special options for the Rose Equation of State loss.
    """

    dx: float = 0.10
    delta: float = 0.01
    crystals: List[str] = None
    beta: List[float] = None


@dataclass
class _HyperParameters:
    """
    The base data class for handling hyper parameters.
    """
    pass


@add_slots
@nested_dataclass
class LossParameters(_HyperParameters):
    """
    Hyper parameters for constructing the total loss.
    """
    energy: EnergyLossOptions = EnergyLossOptions()
    forces: ForcesLossOptions = ForcesLossOptions()
    stress: StressLossOptions = StressLossOptions()
    total_pressure: PressureLossOptions = PressureLossOptions()
    l2: L2LossOptions = L2LossOptions()
    elastic: ElasticLossOptions = ElasticLossOptions()
    rose: RoseLossOptions = RoseLossOptions()


@add_slots
@dataclass
class OptParameters(_HyperParameters):
    """
    Hyper parameters for optimizing the total loss.
    """
    method: str = 'adam'
    learning_rate: float = 0.01
    decay_function: Union[str, None] = None
    decay_rate: float = 0.99
    decay_steps: int = 1000
    staircase: bool = False
    additional_kwargs: Dict = None


@add_slots
@dataclass
class CkptParameters(_HyperParameters):
    """
    Hyper parameters for restoring from a checkpoint.
    """

    checkpoint_filename: Union[str, None, bool] = None
    use_ema_variables: bool = True
    restore_all_variables: bool = True


@add_slots
@nested_dataclass
class TrainParameters(_HyperParameters):
    """
    Hyper parameters for handling the training.
    """
    model_dir: str = "train"
    reset_global_step: bool = True
    batch_size: int = 50
    shuffle: bool = True
    max_checkpoints_to_keep: int = 20
    train_steps: int = 10000
    eval_steps: int = 1000
    summary_steps: int = 100
    log_steps: int = 100
    profile_steps: int = 2000
    ckpt: CkptParameters = CkptParameters()


@dataclass(frozen=True)
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
