# The Roadmap of TensorAlloy

* Author: Xin Chen
* Date: Dec 22, 2018

## Release 1.1

### Major Features

- [ ] Improve the training performance of `forces`.

### Code Refactoring

- [ ] `tensoralloy.nn.basic.BasicNN`: 
    - Merge `minimize_properties` and `predict_properties` because only directly
      exported models need `predict_properties`. Thus it should be set in 
      `BatchDescriptorTransformer.as_transformer()`.
- [ ] `tensoralloy.io.input.reader.InputReader`:
    - Move `nn.predict` to `behler.predict`
    - Remove `dataset.k_max`
    - Add a boolean option: `behler.radial` 

## Release 1.2

### Major Features

- [ ] Improve the training performance of `stress`
- [ ] Implement a tensorflow Op: `cubic_spline`
