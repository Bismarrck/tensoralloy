# TensorAlloy

**TensorAlloy** is a TensorFlow based machine learning framework for metal 
alloys. **TensorAlloy** builds direct computation graph from atomic positions 
to total energy. Thus, atomic forces, virial stress tensor and the second-order 
**Hessian** matrix can be derived by the **AutoGrad** module of TensorFlow 
directly.


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

## 2. Training

There are two training examples:

1. QM7
    * This example starts from the raw `extxyz` file. The first step is building
    the SQLITE3 database.
    * Angular symmetry functions are enabled.  

2. SNAP/Ni-Mo
    * The binary alloy dataset.
    * Only radial symmetry functions are used by default.

### 2.1 Input

[TOML](https://github.com/toml-lang/toml) is the configuration file format used
by TensorAlloy. All the necessary keys are included in the two examples. Default
keys and values can be found in 
[default.toml](tensoralloy/io/input/defaults.toml).

### 2.2 Usage

A command-line program [tensoralloy](tools/tensoralloy) is provided. Add the 
directory [tools](tools) to `PATH` to use this program.

Here are some key commands:

* `tensoralloy build database`: build a database from an `extxyz` file. 
* `tensoralloy run`: run a training experiment from a `toml` input file. 
* `tensoralloy --help`: print the help messages.

### 2.3 Model

After the training, a binary `pb` file (the trained model) will be exported. 

We also provide three pre-trained models:

* Ni: energy, force
* Mo: energy, force, stress
* Ni-Mo: energy, force, stress

## 3. Prediction

To calculate properties of an arbitrary Ni-Mo structure, simply do:

```python
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

## 4. License

This TensorAlloy program is licensed under the Apache License, Version 2.0 
(the "License"); you may not use this file except in compliance with the 
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
