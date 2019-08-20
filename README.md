# TensorAlloy

**TensorAlloy** is a TensorFlow based machine learning framework for metal 
alloys. **TensorAlloy** builds direct computation graph from atomic positions 
to total energy. Thus, atomic forces and the virial stress tensor can be derived 
by the **AutoGrad** module of TensorFlow directly.

## 1. Requirements

* Python>=3.7.0
* TensorFlow==1.13.1
* scikit-learn
* scipy
* numpy
* ase>=3.18.0
* matplotlib>=2.1.0
* toml==0.10.0
* wheel

Anaconda3 can install above packages without pain. However, the performance
of conda-provided tensorflow is not that good. 

Natively compiled TensorFlow, with all CPU features (SSE, AVX, etc.) enabled, 
is strongly recommended. 

**Note:** `[prefix]` indicates the top-level directory where the program is 
unzipped.

## 2. Training

There are two training examples:

1. QM7
    * This example starts from the raw `extxyz` file. The first step is building
    the SQLITE3 database.
    * Angular symmetry functions are enabled.  

2. SNAP/Ni-Mo
    * The binary alloy dataset.
    * Only radial symmetry functions are used by default.

### 2.1 Usage

[TOML](https://github.com/toml-lang/toml) is the configuration file format used
by TensorAlloy. All the necessary keys are included in the two examples. Default
keys and values can be found in 
[default.toml](tensoralloy/io/input/defaults.toml).

A command-line program [tensoralloy](tools/tensoralloy) is provided. Add the 
directory [tools](tools) to `PATH` to use this program:

```bash
export PATH=[prefix]/tensoralloy/tools:$PATH
chmod +x [prefix]/tensoralloy/tools/tensoralloy
```

Here are some key commands:

* `tensoralloy build database [extxyz]`: build a database from an `extxyz` file. 
* `tensoralloy run [input.toml]`: run an experiment from a `toml` input file. 
* `tensoralloy print [logfile]`: print the evaluation results from a logfile.
* `tensoralloy --help`: print the help messages.

The `run.sh` in [qm7](examples/qm7/run.sh) and [snap](examples/snap/run.sh) give
more details about these commands.  

### 2.2 Output

After the training, a binary `pb` file (the trained model) will be exported to 
the `model_dir` specified in the input toml file. This exported `pb` file can be
used by the ASE-style [`TensorAlloyCalculator`](tensoralloy/calculator.py). 

We also provide three pre-trained models:

* Ni: energy, force
* Mo: energy, force, stress
* Ni-Mo: energy, force, stress

## 3. Prediction

To use the [`TensorAlloyCalculator`](tensoralloy/calculator.py) calculator, 
`PYTHONPATH` should be configured first:

```bash
export PYTHONPATH=[prefix]:${PYTHONPATH}
```

To calculate properties of an arbitrary Ni-Mo structure, simply do:

```python
#!coding=utf-8
""" Simple usage. """
from tensoralloy.calculator import TensorAlloyCalculator
from ase.build import bulk
from ase.units import GPa

calc = TensorAlloyCalculator("NiMo.pb")
atoms = bulk("Ni", cubic=True)
atoms.calc = calc
print(atoms.get_total_energy())
print(atoms.get_forces())
print(atoms.get_stress() / GPa)
```

## 4. Input

In this section we will introduce the options and values of the input toml file.

### 4.1 Root

* `precision`: `medium` (float32) or `high` (float64). Default is `medium`.
* `seed`: the global seed. Default is 611.

### 4.2 Dataset

* `dataset.sqlite3`
* `dataset.name`
* `dataset.rc`
* `dataset.tfrecords_dir`
* `dataset.test_size`
* `dataset.serial`

### 4.3 NN

* `nn.activation`
* `nn.minimize`
* `nn.export`

* `nn.loss.energy.weight`
* `nn.loss.energy.per_atom_loss`
* `nn.loss.energy.method`

* `nn.loss.forces.weight`
* `nn.loss.forces.method`

* `nn.loss.stress.weight`
* `nn.loss.stress.method`

* `nn.loss.l2.weight`
* `nn.loss.l2.decayed`
* `nn.loss.l2.decay_rate`
* `nn.loss.l2.decay_steps`

* `nn.atomic.arch`
* `nn.atomic.kernel_initializer`
* `nn.atomic.minmax_scale`

* `nn.atomic.resnet.fixed_static_energy`

* `nn.atomic.behler.eta`
* `nn.atomic.behler.omega`
* `nn.atomic.behler.beta`
* `nn.atomic.behler.gamma`
* `nn.atomic.behler.zeta`
* `nn.atomic.behler.angular`

#### 4.3.1 Layers



### 4.4 Opt

* `opt.method`
* `opt.learning_rate`
* `opt.decay_function`
* `opt.decay_rate`
* `opt.decay_steps`
* `opt.staircase`

### 4.5 Train

* `train.reset_global_step`
* `train.batch_size`
* `train.shuffle`
* `train.model_dir`
* `train.train_steps`
* `train.eval_steps`
* `train.summary_steps`
* `train.log_steps`
* `train.profile_steps`

* `train.ckpt.checkpoint_filename`
* `train.ckpt.use_ema_variables`
* `train.ckpt.restore_all_variables`

## 5. License

This TensorAlloy program is licensed under the Apache License, Version 2.0 
(the "License"); you may not use this file except in compliance with the 
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
