# coding=utf-8
"""
This module defines a TOML based input reader.
"""
from __future__ import print_function, absolute_import

import toml
from os.path import dirname, join
from typing import List, Union, Any

from tensoralloy.utils import get_elements_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def nested_get(d: dict, nested_keys: Union[str, List[str]]) -> Any:
    """
    Get the value from the dict `d` with a keypath (e.g `a.b.c`) or a list of
    nested keys (e.g ['a', 'b', 'c']).

    Parameters
    ----------
    d : dict
        A dict.
    nested_keys : str or List[str]
        A str as the key path or a list of str.

    Returns
    -------
    val : Any
        The value corresponding to the keypath.

    """
    if isinstance(nested_keys, str):
        nested_keys = nested_keys.split('.')
    obj = d
    for i, key in enumerate(nested_keys):
        if not hasattr(obj, "__getitem__"):
            return None
        obj = obj.get(key, None)
        if obj is None:
            return None
    return obj


def nested_set(d: dict, nested_keys: Union[str, List[str]], new_val):
    """
    Set the value of dict `d` with the given keypath or nested keys.
    """
    if isinstance(nested_keys, str):
        nested_keys = nested_keys.split('.')
    if not isinstance(d, dict):
        raise ValueError("`d` must be a dict")
    obj = d
    n = len(nested_keys)
    for i, key in enumerate(nested_keys):
        if i == n - 1:
            obj[key] = new_val
        else:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]


class InputValueError(ValueError):
    """
    Inappropriate argument value (of correct type).
    """

    def __init__(self, keypath: Union[str, List[str]], value, valid_options):
        if isinstance(keypath, (list, tuple)):
            keypath = '.'.join(map(str, keypath))
        self._keypath = keypath
        self._value = value
        self._valid_options = valid_options

    def __str__(self):
        error = f"{self._value} is not a valid option for [{self._keypath}]"
        desc = f"Supported options: {', '.join(map(str, self._valid_options))}"
        return f"{error}. {desc}"


class InputReader:
    """
    A TOML based input reader.
    """

    def __init__(self, filename):
        """
        Initialization method.

        Parameters
        ----------
        filename : str
            The input file to read.

        """
        with open(join(dirname(__file__), "defaults.toml")) as fp:
            defaults = toml.load(fp)

        with open(join(dirname(__file__), "choices.toml")) as fp:
            choices = toml.load(fp)

        with open(filename) as fp:
            configs = toml.load(fp)

        self._configs = self._merge(defaults, configs, choices)

    @property
    def configs(self):
        """
        Return the dict of configs.
        """
        return self._configs

    def __getitem__(self, keypath):
        return nested_get(self._configs, keypath)

    @staticmethod
    def _merge(defaults: dict, configs: dict, choices: dict):
        """
        Merge the parsed and the default configs.
        """
        results = defaults.copy()

        def _safe_set(_keypath, _val):
            _options = nested_get(choices, _keypath)
            if _options is not None:
                if _val not in _options:
                    if not isinstance(_val, bool):
                        raise InputValueError(_keypath, _val, _options)
                    else:
                        _val = None
            nested_set(results, _keypath, _val)

        def _safe_update(_keypath, _dst=None, required=False):
            _new_val = nested_get(configs, _keypath)
            if _new_val is None:
                if not required:
                    return
                else:
                    raise ValueError(f"{_keypath} should be provided")
            if _dst is None:
                _dst = _keypath
            _safe_set(_dst, _new_val)

        for section in ("dataset", "nn", "opt", "train"):
            for key, val in defaults[section].items():
                if isinstance(val, dict):
                    continue
                _safe_update(f"{section}.{key}", required=(val == 'required'))

        if nested_get(results, 'dataset.name').find("-") >= 0:
            raise ValueError("'-' is not allowed in 'dataset.name'.")
        
        _safe_update("nn.loss.weight")

        descriptor = nested_get(configs, 'dataset.descriptor')

        if descriptor == 'behler':
            _safe_update("nn.atomic.behler.angular")
            _safe_update("nn.atomic.behler.eta")
            _safe_update("nn.atomic.behler.gamma")
            _safe_update("nn.atomic.behler.beta")
            _safe_update("nn.atomic.behler.zeta")
            _safe_update("nn.atomic.arch")
            _safe_update("nn.atomic.input_normalizer")
            _safe_update("nn.atomic.export")

            layers = nested_get(configs, 'nn.atomic.layers')
            if isinstance(layers, dict):
                for key, val in layers.items():
                    nested_set(results, f'nn.atomic.layers.{key}', val)

            if 'eam' in results['nn']:
                del results['nn']['eam']

        else:
            _safe_update("nn.eam.arch")
            _safe_update("nn.eam.export.nr")
            _safe_update("nn.eam.export.dr")
            _safe_update("nn.eam.export.nrho")
            _safe_update("nn.eam.export.drho")
            _safe_update("nn.eam.export.checkpoint")

            for attr in ('constant', 'type'):
                values = nested_get(configs, f"nn.eam.export.lattice.{attr}")
                if isinstance(values, dict):
                    for element in values.keys():
                        _safe_update(f"nn.eam.export.lattice.{attr}.{element}")

            for func in ('embed', 'phi', 'rho'):
                pots = nested_get(configs, f"nn.eam.{func}")
                if isinstance(pots, dict):
                    for key in pots.keys():
                        src = f"nn.eam.{func}.{key}"
                        if func == 'phi':
                            key = "".join(
                                sorted(get_elements_from_kbody_term(key)))
                            dst = f"nn.eam.{func}.{key}"
                        else:
                            dst = None
                        _safe_update(src, dst)

            if 'behler' in results:
                del results['behler']
            if 'atomic' in results['nn']:
                del results['nn']['atomic']

        return results
