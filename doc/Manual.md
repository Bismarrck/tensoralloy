# TensorAlloy Manual

* Author: Xin Chen
* Email: Bismarrck@me.com
* Version: `1.0`
* Date: Dec 22, 2018

## 1. Program

The `tensoralloy` program has two subcommands:

* `build`: build a `sqlite3` database from a `xyz` or `extxyz` file.
* `run`: run an experiment from a TOML input file.

### 1.1 Subcommand: build

The `build` comamnd is used to build a 
[ase.db](https://wiki.fysik.dtu.dk/ase/ase/db/db.html#module-ase.db.core).

**Usage**

```bash
tensoralloy build filename [-h] 
                           [-n NUM_EXAMPLES] 
                           [--energy-unit ENERGY_UNIT] 
                           [--forces-unit FORCES_UNIT] 
                           [--stress-unit STRESS_UNIT]
```

**Args:**

* `filename`: a `xyz` file or a `extxyz` file.
* `-n`: the total number of examples to read. If not set, all will be read.
* `--energy-unit`: the unit of the total energies, possible values:
    - eV (default)
    - Hartree
    - kcal/mol
* `--forces-unit`: the unit of the atomic forces, possible values:
    - eV/Angstrom (default)
    - eV/Bohr
    - Hartree/Angstrom
    - Hartree/Bohr
    - kcal/mol/Angstrom
    - kcal/mol/Bohr
* `--stress-unit`: the unit of the stress tensors, possible values:
    - GPa (default)
    - kbar

**Example**

```bash
tensoralloy build datasets/qm7.xyz --energy-unit=Hartree
```

### 1.2 Subcommand: run

The subcommand `run` is used to run an experiment given a 
[TOML](https://github.com/toml-lang/toml) input file. Section 2 will introduce 
the input file. 

**Usage**

```bash
tensoralloy run input_file
```

**Args**

* input_file: a TOML input file.

**Example**

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
cd test_files/inputs
tensoralloy run qm7.behler.k2.toml
```

## 2. Input
