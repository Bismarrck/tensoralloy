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


@dataclass
class EnergyOp:
    """
    An energy op. `total` is the total energy (scalar) while `atomic` is a
    vector. `total = sum(atomic)`
    """
    total: tf.Tensor
    atomic: tf.Tensor


@dataclass
class EnergyOps:
    """
    Different types of energy ops.

    energy: the internal energy U

    """
    energy: EnergyOp

    def as_dict(self):
        """ Dict representation of the ops. """
        return {"energy": self.energy.total, "energy/atom": self.energy.atomic}


@dataclass
class FiniteTemperatureEnergyOps(EnergyOps):
    """
    Energy ops for finite temperature potentials.

    eentropy: the electron entropy S
    free_energy: electron free energy F = U - T*S

    """
    eentropy: EnergyOp
    free_energy: EnergyOp

    def as_dict(self):
        """ Dict representation of the ops. """
        adict = super(FiniteTemperatureEnergyOps, self).as_dict()
        adict.update({
            "free_energy": self.free_energy.total,
            "free_energy/atom": self.free_energy.atomic,
            "eentropy": self.eentropy.total,
            "eentropy/atom": self.eentropy.atomic})
        return adict


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
    Special options for the loss of energy (internal energy).
    """
    per_atom_loss: bool = False
    method: str = 'rmse'


@add_slots
@dataclass
class EEntropyLossOptions(EnergyLossOptions):
    """
    Special options for the loss of electron entropy.
    """
    pass


@add_slots
@dataclass
class FreeEnergyLossOptions(EnergyLossOptions):
    """
    Special options for the loss of free energy.
    """
    pass


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
    xlo: float = 0.90
    xhi: float = 1.10
    dx: float = 0.01
    crystals: List[str] = None
    beta: List[float] = None
    p_target: List[float] = None
    E_target: List[float] = None


@add_slots
@dataclass
class EEntropyConstraintOptions(_LossOptions):
    """
    Special options for the eentropy constraint loss.
    """
    crystals: List[str] = None


@add_slots
@dataclass
class ExtraDBConstraintOptions(_LossOptions):
    """
    The extra database constraint.
    """
    filename: str = None
    minimize: List[str] = None

@add_slots
@dataclass
class EnergyDifferenceLossOptions(_LossOptions):
    """
    Special options for the energy difference loss.
    """
    references: List[str] = None
    crystals: List[str] = None
    diff: List[float] = None
    forces_weight: float = 1.0
    method: str = "mae"


@add_slots
@dataclass
class ForceConstantsLossOptions(_LossOptions):
    """
    Special options for force constants constraint loss.
    """
    crystals: List[str] = None
    forces_weight: float = 1.0


@dataclass
class _HyperParameters:
    """
    The base data class for handling hyper parameters.
    """
    pass


class AdaptiveSampleWeightOptions():
    """
    Options for using the adaptive sample weight scheme.
    """
    enabled: bool = False
    method: str = "sigmoid"
    params: List[float] = None


@add_slots
@nested_dataclass
@dataclass
class LossParameters(_HyperParameters):
    """
    Hyper parameters for constructing the total loss.
    """
    energy: EnergyLossOptions = EnergyLossOptions()
    eentropy: EEntropyLossOptions = EEntropyLossOptions()
    free_energy: FreeEnergyLossOptions = FreeEnergyLossOptions()
    forces: ForcesLossOptions = ForcesLossOptions()
    stress: StressLossOptions = StressLossOptions()
    total_pressure: PressureLossOptions = PressureLossOptions()
    l2: L2LossOptions = L2LossOptions()
    elastic: ElasticLossOptions = ElasticLossOptions()
    rose: RoseLossOptions = RoseLossOptions()
    ediff: EnergyDifferenceLossOptions = EnergyDifferenceLossOptions()
    eentropy_constraint: EEntropyConstraintOptions = EEntropyConstraintOptions()
    hessian_constraint: ForceConstantsLossOptions = ForceConstantsLossOptions()
    extra_constraint: ExtraDBConstraintOptions = ExtraDBConstraintOptions()
    adaptive_sample_weight: AdaptiveSampleWeightOptions = AdaptiveSampleWeightOptions()

    def __getitem__(self, item):
        return getattr(self, item)


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
    restore_optimizer_variables: bool = True


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
