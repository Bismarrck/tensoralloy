#!coding=utf-8
"""
This module defines data classes for `tensoralloy.nn` package.
"""
from __future__ import print_function, absolute_import

from dataclasses import dataclass, fields, is_dataclass
from typing import List, Union

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def add_slots(cls):
    """
    A decorator to put __slots__ on a dataclass with fields with defaults.

    Need to create a new class, since we can't set __slots__ after a class has
    been created.

    References
    ----------
    https://github.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py

    """
    # Make sure __slots__ isn't already set.
    if '__slots__' in cls.__dict__:
        raise TypeError(f'{cls.__name__} already specifies __slots__')

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in fields(cls))
    cls_dict['__slots__'] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop('__dict__', None)
    # And finally create the class.
    qualname = getattr(cls, '__qualname__', None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls


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


@add_slots
@dataclass
class _LossOptions:
    """
    The basic options for a loss.
    """
    weight: float = 1.0


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
    No extra option for the loss of forces.
    """
    pass


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


@add_slots
@dataclass
class ElasticConstraintOptions:
    """
    Options for computing loss of the elastic contraints.
    """

    use_kbar: bool = True
    forces_weight: float = 1.0
    stress_weight: float = 0.1


@add_slots
@nested_dataclass
@dataclass
class ElasticLossOptions(_LossOptions):
    """
    Special options for the loss of elastic constants.
    """

    crystals: List[str] = None
    constraint: ElasticConstraintOptions = ElasticConstraintOptions()


@dataclass
class _HyperParameters:
    """
    The base data class for handling hyper parameters.
    """
    pass


@add_slots
@nested_dataclass
@dataclass
class LossParameters(_HyperParameters):
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


@add_slots
@dataclass
class TrainParameters(_HyperParameters):
    """
    Hyper parameters for handling the training.
    """

    model_dir: str = "train"
    restart: bool = True
    batch_size: int = 50
    previous_checkpoint: Union[str, None, bool] = None
    shuffle: bool = True
    max_checkpoints_to_keep: int = 20
    train_steps: int = 10000
    eval_steps: int = 1000
    summary_steps: int = 100
    log_steps: int = 100
    profile_steps: int = 2000


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
