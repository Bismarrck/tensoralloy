# TensorAlloy

* Developer: Xin Chen, De-Ye Lin, Hai-Feng Song
* Contact: Bismarrck@me.com

**TensorAlloy** is a TensorFlow based machine learning framework for metal 
alloys. **TensorAlloy** builds direct computation graph from atomic positions 
to total energy:

Thus, atomic forces and the virial stress tensor can be derived by the 
**AutoGrad** module of TensorFlow directly:

```python
forces = tf.gradients(E, R)[0]
stress = -0.5 * (tf.gradients(E, h)[0] @ h)
```

where `E` is the total energy tensor built from atomic positions `R` and `h` is 
the 3x3 cell tensor.


## 1. Requirements

* Python>=3.7.0
* TensorFlow>=1.11
* scikit-learn
* scipy
* numpy
* ase>=3.15.0
* atsim.potentials==0.2.1
* matplotlib>=2.1.0

Anaconda3 can install above packages without pain. For performance reason 
natively compiled TensorFlow, with all CPU features (SSE, AVX, etc.) enabled, is 
strongly recommended. 

## 2. Compilation

Run the bash script [`build_dist.sh`](./build_dist.sh) to compile this package. 
An **egg** file, `tensoralloy-1.0.0-py3.7-{platform-identifier}.egg`, will be 
generated under `dist`. This egg can be installed with 
[`easy_install`](https://setuptools.readthedocs.io/en/latest/easy_install.html).
