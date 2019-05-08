# coding=utf-8
"""
This module defines a TOML based input reader.
"""
from __future__ import print_function, absolute_import

import toml

from os.path import dirname, join
from typing import List, Union

from tensoralloy.utils import get_elements_from_kbody_term
from tensoralloy.utils import nested_get, nested_set

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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
        input_dir = dirname(filename)

        with open(join(dirname(__file__), "defaults.toml")) as fp:
            defaults = toml.load(fp)

        with open(join(dirname(__file__), "choices.toml")) as fp:
            choices = toml.load(fp)

        with open(filename) as fp:
            configs = toml.load(fp)

        self._configs = self._merge(defaults, configs, choices, input_dir)

    @property
    def configs(self):
        """
        Return the dict of configs.
        """
        return self._configs

    def __str__(self):
        """
        Return the string representation of this input file.
        """
        return toml.dumps(self._configs)

    def __getitem__(self, keypath):
        return nested_get(self._configs, keypath)

    def __setitem__(self, keypath, value):
        nested_set(self._configs, keypath, value)

    @staticmethod
    def _merge(defaults: dict, configs: dict, choices: dict, input_dir: str):
        """
        Merge the parsed and the default configs.

        Parameters
        ----------
        defaults : dict
            The default parameters and values.
        configs : dict
            The parameters and corresponding values parsed from the input file.
        choices : dict
            Restricted choices of certain parameters.
        input_dir : str
            The directory where the input file is.

        """
        results = defaults.copy()

        def _safe_set(_keypath, _val):
            _options = nested_get(choices, _keypath)
            if isinstance(_options, dict):
                for _subkey, _subval in _options.items():
                    if _subkey in _val:
                        _safe_set(f'{_keypath}.{_subkey}', _val[_subkey])
            elif _options is not None:
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

        def _convert_filepath(_filepath: str):
            if _filepath.startswith('/'):
                # Absolute path. No change.
                return _filepath
            else:
                return join(input_dir, _filepath)

        _safe_update('precision')
        _safe_update('seed')
        
        for section in ("dataset", "nn", "opt", "train", "debug"):
            for key, val in defaults[section].items():
                if isinstance(val, dict):
                    continue
                _safe_update(f"{section}.{key}", required=(val == 'required'))

        _safe_update("nn.loss")
        _safe_update("nn.loss.energy")
        _safe_update("nn.loss.forces")
        _safe_update("nn.loss.stress")
        _safe_update("nn.loss.total_pressure")
        _safe_update("nn.loss.l2")
        _safe_update("nn.loss.elastic")

        if nested_get(results, 'dataset.name').find("-") >= 0:
            raise ValueError("'-' is not allowed in 'dataset.name'.")

        descriptor = nested_get(configs, 'dataset.descriptor')

        if descriptor == 'behler':
            _safe_update("nn.atomic.behler")
            _safe_update("nn.atomic.arch")
            _safe_update("nn.atomic.resnet")

            layers = nested_get(configs, 'nn.atomic.layers')
            if isinstance(layers, dict):
                for key, val in layers.items():
                    nested_set(results, f'nn.atomic.layers.{key}', val)

            if 'eam' in results['nn']:
                del results['nn']['eam']

        else:
            _safe_update("nn.eam.arch")
            _safe_update("nn.eam.setfl")

            for attr in ('constant', 'type'):
                values = nested_get(configs, f"nn.eam.setfl.lattice.{attr}")
                if isinstance(values, dict):
                    for element in values.keys():
                        _safe_update(f"nn.eam.setfl.lattice.{attr}.{element}")

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

        # Convert the paths
        for keypath in ("dataset.sqlite3",
                        "dataset.tfrecords_dir",
                        "train.model_dir",
                        "train.previous_checkpoint"):
            filepath = nested_get(results, keypath)
            if keypath == "train.previous_checkpoint" and filepath is False:
                continue
            nested_set(results, keypath, _convert_filepath(filepath))

        return results
