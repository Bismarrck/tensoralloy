#!coding=utf-8
"""
Data classes for `tensoralloy.training`.
"""
from __future__ import print_function, absolute_import

from dataclasses import dataclass

from tensoralloy.nn.dataclasses import TrainParameters, OptParameters
from tensoralloy.nn.dataclasses import LossParameters
from tensoralloy.io.input.reader import InputReader

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass(frozen=True)
class DebugParameters:
    """
    Parameters for debugging.
    """
    logging_level: str = 'info'
    start_delay_secs: int = 300
    throttle_secs: int = 60
    allow_gpu_growth: bool = True


@dataclass(frozen=True)
class DistributeParameters:
    """
    Parameters for distributed training.
    """
    strategy: str = "default"
    num_workers: int = 1
    all_reduce_alg: str = "auto"
    num_packs: int = 1
    num_gpus: int = 0

    @property
    def num_replicas(self):
        """
        Return the number of replicas.
        """
        return max(self.num_gpus, 1)

    def as_dict(self):
        """
        Return the dict representation.
        """
        alg = self.all_reduce_alg if self.all_reduce_alg != "auto" else None
        return {"distribution_strategy": self.strategy,
                "num_gpus": self.num_gpus,
                "num_packs": self.num_packs,
                "num_workers": self.num_workers,
                "all_reduce_alg": alg}


class EstimatorHyperParams(dict):
    """
    Hyper parameters for the estimator.
    """

    @property
    def train(self) -> TrainParameters:
        """
        Return training parameters.
        """
        return self["train"]

    @property
    def debug(self) -> DebugParameters:
        """
        Return debugging parameters.
        """
        return self["debug"]

    @property
    def opt(self) -> OptParameters:
        """
        Return optimation parameters.
        """
        return self["opt"]

    @property
    def loss(self) -> LossParameters:
        """
        Return loss parameters.
        """
        return self["loss"]

    @property
    def distribute(self) -> DistributeParameters:
        """
        Return parameters for distributed training.
        """
        return self["distribute"]

    @property
    def seed(self) -> int:
        """
        Return the seed.
        """
        return self["seed"]

    @property
    def precision(self) -> str:
        """
        Return the float precision.
        """
        return self['precision']

    @classmethod
    def from_input_reader(cls, reader: InputReader):
        """
        Initialize an `EstimatorHyperParams` from an `InputReader`.
        """
        opt_dict = reader['opt']
        method = opt_dict['method']
        minimizer_kwargs = opt_dict.pop(method)
        opt_parameters = OptParameters(additional_kwargs=minimizer_kwargs,
                                       **opt_dict)

        return cls(
            seed=reader['seed'],
            precision=reader['precision'],
            train=TrainParameters(**reader['train']),
            opt=opt_parameters,
            loss=LossParameters(**reader['nn.loss']),
            debug=DebugParameters(**reader['debug']),
            distribute=DistributeParameters(**reader["distribute"])
        )
