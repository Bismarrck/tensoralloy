# coding=utf-8
"""
This module defines unit tests of `tensoralloy.io.input.InputReader`.
"""
from __future__ import print_function, absolute_import

import nose

from nose.tools import assert_equal, assert_list_equal
from nose.tools import assert_not_in
from os.path import join, realpath

from tensoralloy.io.input.reader import InputReader
from tensoralloy.utils import nested_get
from tensoralloy.test_utils import test_dir, datasets_dir, project_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_read_sf_angular_toml():
    """
    Test `InputReader` with file 'qm7.sfa.toml'.
    """
    reader = InputReader(join(test_dir(), 'inputs', 'qm7.sfa.toml'))
    configs = reader.configs

    assert_equal(reader['precision'], 'medium')
    assert_equal(reader['nn.atomic.sf.trainable'], True)

    assert_equal(realpath(reader['dataset.sqlite3']),
                 realpath(join(datasets_dir(True), 'qm7.db')))
    assert_equal(realpath(reader['dataset.tfrecords_dir']),
                 realpath(join(project_dir(True), 'experiments/qm7-k3')))
    assert_equal(realpath(reader['train.model_dir']),
                 realpath(join(project_dir(True), 'experiments/qm7-k3/train')))
    assert_equal(reader['train.ckpt.checkpoint_filename'], False)

    assert_equal(nested_get(configs, 'pair_style'), 'atomic/sf')
    assert_equal(nested_get(configs, 'nn.atomic.kernel_initializer'),
                 'truncated_normal')
    assert_list_equal(nested_get(configs, 'nn.atomic.sf.eta'),
                      [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 20.0, 40.0])
    assert_list_equal(reader['nn.atomic.sf.omega'],
                      [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert_equal(nested_get(configs, 'nn.atomic.sf.angular'), True)
    assert_equal(nested_get(configs, 'nn.atomic.sf.cutoff_function'),
                 'polynomial')
    assert_not_in('eam', configs['nn'])
    assert_list_equal(nested_get(configs, 'nn.atomic.layers.C'), [64, 32])
    assert_list_equal(nested_get(configs, 'nn.atomic.layers.H'), [64, 32])
    assert_list_equal(nested_get(configs, 'nn.atomic.layers.N'), [64, 32])
    assert_list_equal(reader['nn.minimize'], ['energy'])
    assert_list_equal(reader['nn.export'], ['energy', 'forces', 'hessian'])

    assert_equal(reader["distribute.num_gpus"], 4)
    assert_equal(reader["distribute.num_packs"], 1)
    assert_equal(reader["distribute.all_reduce_alg"], "auto")
    assert_equal(reader["distribute.strategy"], "mirrored")


def test_read_eam_alloy_toml():
    """
    Test `InputReader` with file 'qm7.alloy.eam.toml' for an `eam/alloy` task.
    """
    reader = InputReader(join(test_dir(), 'inputs', 'snap_Ni.zjw04.toml'))
    configs = reader.configs

    assert_equal(reader['precision'], 'medium')

    assert_equal(realpath(reader['train.model_dir']),
                 realpath(join(test_dir(True), "inputs", "snap_Ni_zjw04")))

    assert_not_in('atomic', configs['nn'])
    assert_equal(len(nested_get(configs, 'nn.eam.rho')), 1)
    assert_equal(len(nested_get(configs, 'nn.eam.embed')), 1)
    assert_list_equal(reader['nn.minimize'], ['energy', 'forces', 'rose'])
    assert_list_equal(reader['nn.export'],
                      ['energy', 'forces', 'stress', 'hessian'])
    assert_equal(reader['nn.loss.l2.weight'], 0.01)
    assert_equal(reader['nn.loss.forces.weight'], 3.0)
    assert_equal(reader['nn.loss.energy.weight'], 1.0)
    assert_list_equal(reader['nn.loss.rose.crystals'], ['Ni'])
    assert_equal(reader['nn.loss.rose.beta'][0], 0.5e-2)

    assert_equal(reader['opt.method'], 'sgd')
    assert_equal(reader['opt.sgd.momentum'], 0.9)
    assert_equal(reader['opt.sgd.use_nesterov'], False)


def test_read_eam_fs_toml():
    """
    Test `InputReader` with file 'AlFe.fs.eam.toml' for an `eam/fs` task.
    """
    reader = InputReader(join(test_dir(), 'inputs', 'AlFe.fs.eam.toml'))
    configs = reader.configs

    assert_equal(nested_get(configs, 'train.batch_size'), 50)
    assert_equal(nested_get(configs, 'train.shuffle'), True)
    assert_list_equal(reader['nn.export'],
                      ['energy', 'forces', 'hessian', 'stress'])
    assert_list_equal(reader['nn.minimize'],
                      ['energy', 'forces', 'stress', 'elastic'])
    assert_equal(reader['nn.loss.stress.method'], 'logcosh')

    assert_equal(realpath(reader['nn.loss.elastic.crystals'][0]),
                 realpath(join(test_dir(True), "inputs", "Al.toml")))
    assert_equal(reader['nn.loss.elastic.weight'], 0.1)
    assert_equal(reader['nn.loss.elastic.constraint.use_kbar'], False)
    assert_equal(reader['nn.loss.elastic.constraint.forces_weight'], 1.0)
    assert_equal(reader['nn.loss.elastic.constraint.stress_weight'], 0.01)

    assert_equal(nested_get(configs, 'nn.eam.setfl.lattice.type.Al'), 'fcc')
    assert_equal(nested_get(configs, 'nn.eam.setfl.lattice.type.Fe'), 'bcc')
    assert_equal(len(nested_get(configs, 'nn.eam.rho')), 4)
    assert_equal(len(nested_get(configs, 'nn.eam.embed')), 2)
    assert_list_equal(nested_get(configs, 'nn.eam.phi.AlFe'), [32, 32])
    assert_equal(reader['nn.eam.phi.AlAl'], 'msah11')
    assert_equal(reader['train.ckpt.checkpoint_filename'], False)


def test_read_deepmd_toml():
    """
    Test `InputReader` with the file `Ni.deepmd.toml` for a `DeePotSE` task.
    """
    reader = InputReader(join(test_dir(), 'inputs', 'Ni.deepmd.toml'))

    assert_equal(reader['pair_style'], 'atomic/deepmd')
    assert_equal(reader['nn.atomic.deepmd.m1'], 50)
    assert_equal(reader['nn.atomic.deepmd.m2'], 4)
    assert_list_equal(reader['nn.atomic.deepmd.embedding_sizes'], [20, 40])


def test_read_grap_toml():
    """
    Test `InputReader` with the file `Be_grap_sf_quad.toml` for a `GRAP` task.
    """
    reader = InputReader(join(test_dir(), 'inputs', 'Be_grap_sf_quad.toml'))

    assert_equal(reader['pair_style'], 'atomic/grap')
    assert_list_equal(reader['nn.atomic.grap.sf.omega'], [0.0, 1.5, 3.0])
    assert_equal(reader['nn.atomic.grap.multipole'], 2)
    assert_equal(reader['nn.atomic.grap.algorithm'], 'sf')


if __name__ == "__main__":
    nose.run()
