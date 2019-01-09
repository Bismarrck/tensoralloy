# coding=utf-8
"""
This module defines unit tests of `EamNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import os
import shutil

from nose.tools import assert_equal, assert_tuple_equal, assert_almost_equal
from nose.tools import assert_dict_equal, assert_list_equal, with_setup
from os.path import join, exists
from os import remove
from collections import Counter
from unittest import skipUnless
from ase.calculators.lammpsrun import LAMMPS
from ase.build import bulk

from tensoralloy.nn.eam.alloy import EamAlloyNN
from tensoralloy.transformer import EAMTransformer
from tensoralloy.misc import AttributeDict, test_dir, skip, Defaults
from tensoralloy.test_utils import assert_array_equal
from tensoralloy.nn.utils import GraphKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AlCuData:
    """
    A private data container for unit tests of this module.
    """

    def __init__(self):
        """
        Initialization method.
        """
        self.batch_size = 10
        self.max_n_terms = 2
        self.max_n_al = 5
        self.max_n_cu = 7
        self.max_n_atoms = self.max_n_al + self.max_n_cu + 1
        self.max_occurs = Counter({'Al': self.max_n_al, 'Cu': self.max_n_cu})
        self.nnl = 10
        self.elements = sorted(['Cu', 'Al'])

        shape_al = (self.batch_size, self.max_n_terms, self.max_n_al, self.nnl)
        shape_cu = (self.batch_size, self.max_n_terms, self.max_n_cu, self.nnl)

        self.g_al = np.random.randn(*shape_al)
        self.m_al = np.random.randint(0, 2, shape_al).astype(np.float64)
        self.g_cu = np.random.randn(*shape_cu)
        self.m_cu = np.random.randint(0, 2, shape_cu).astype(np.float64)

        self.y_alal = np.random.randn(self.batch_size, self.max_n_al)
        self.y_alcu = np.random.randn(self.batch_size, self.max_n_al)
        self.y_cucu = np.random.randn(self.batch_size, self.max_n_cu)
        self.y_cual = np.random.randn(self.batch_size, self.max_n_cu)

        with tf.name_scope("Inputs"):
            self.descriptors = AttributeDict(
                Al=(tf.convert_to_tensor(self.g_al, tf.float64, 'g_al'),
                    tf.convert_to_tensor(self.m_al, tf.float64, 'm_al')),
                Cu=(tf.convert_to_tensor(self.g_cu, tf.float64, 'g_cu'),
                    tf.convert_to_tensor(self.m_cu, tf.float64, 'm_cu')))
            self.positions = tf.constant(
                np.random.rand(1, self.max_n_atoms, 3),
                dtype=tf.float64,
                name='positions')
            self.mask = np.ones((self.batch_size, self.max_n_atoms), np.float64)
            self.features = AttributeDict(
                descriptors=self.descriptors, positions=self.positions,
                mask=self.mask)
            self.atomic_splits = AttributeDict(
                AlAl=tf.convert_to_tensor(self.y_alal, tf.float64, 'y_alal'),
                AlCu=tf.convert_to_tensor(self.y_alcu, tf.float64, 'y_alcu'),
                CuCu=tf.convert_to_tensor(self.y_cucu, tf.float64, 'y_cucu'),
                CuAl=tf.convert_to_tensor(self.y_cual, tf.float64, 'y_cual'))
            self.symmetric_atomic_splits = AttributeDict(
                AlAl=tf.convert_to_tensor(self.y_alal, tf.float64, 'y_alal'),
                CuCu=tf.convert_to_tensor(self.y_cucu, tf.float64, 'y_cucu'),
                AlCu=tf.convert_to_tensor(
                    np.concatenate((self.y_alcu, self.y_cual), axis=1),
                    tf.float64, 'y_alcu'))


def test_dynamic_stitch_2el():
    """
    Test the method `EamNN._dynamic_stitch` with two types of element.
    """
    with tf.Graph().as_default():

        data = AlCuData()
        max_occurs = data.max_occurs

        nn = EamAlloyNN(elements=data.elements, minimize_properties=['energy'],
                        export_properties=['energy'])

        op = nn._dynamic_stitch(data.atomic_splits, max_occurs, symmetric=False)
        symm_op = nn._dynamic_stitch(data.symmetric_atomic_splits, max_occurs,
                                     symmetric=True)

        ref = np.concatenate((data.y_alal + data.y_alcu,
                              data.y_cucu + data.y_cual),
                             axis=1)

        with tf.Session() as sess:
            results = sess.run([op, symm_op])

            assert_tuple_equal(results[0].shape, ref.shape)
            assert_tuple_equal(results[1].shape, ref.shape)

            assert_array_equal(results[0], ref)
            assert_array_equal(results[1], ref)


@skip
def test_dynamic_stitch_3el():
    """
    Test the method `EamNN._dynamic_stitch` with three types of element.
    """
    with tf.Graph().as_default():

        batch_size = 10
        max_n_al = 5
        max_n_cu = 7
        max_n_mg = 6
        max_occurs = Counter({'Al': max_n_al, 'Mg': max_n_mg, 'Cu': max_n_cu})

        y_alal = np.random.randn(batch_size, max_n_al)
        y_alcu = np.random.randn(batch_size, max_n_al)
        y_almg = np.random.randn(batch_size, max_n_al)
        y_cual = np.random.randn(batch_size, max_n_cu)
        y_cucu = np.random.randn(batch_size, max_n_cu)
        y_cumg = np.random.randn(batch_size, max_n_cu)
        y_mgal = np.random.randn(batch_size, max_n_mg)
        y_mgcu = np.random.randn(batch_size, max_n_mg)
        y_mgmg = np.random.randn(batch_size, max_n_mg)

        symmetric_partitions = AttributeDict(
            AlAl=tf.convert_to_tensor(y_alal, tf.float64, 'y_alal'),
            MgMg=tf.convert_to_tensor(y_mgmg, tf.float64, 'y_mgmg'),
            CuCu=tf.convert_to_tensor(y_cucu, tf.float64, 'y_cucu'),
            AlCu=tf.convert_to_tensor(
                np.concatenate((y_alcu, y_cual), axis=1), name='y_alcu'),
            AlMg=tf.convert_to_tensor(
                np.concatenate((y_almg, y_mgal), axis=1), name='y_almg'),
            CuMg=tf.convert_to_tensor(
                np.concatenate((y_cumg, y_mgcu), axis=1), name='y_cumg'),
        )

        nn = EamAlloyNN(elements=['Al', 'Cu', 'Mg'])
        op = nn._dynamic_stitch(
            symmetric_partitions, max_occurs, symmetric=True)

        ref = np.concatenate(
            (y_alal + y_alcu + y_almg,
             y_cual + y_cucu + y_cumg,
             y_mgal + y_mgcu + y_mgmg),
            axis=1)

        with tf.Session() as sess:
            results = sess.run(op)
            assert_array_equal(results, ref)


@skip
def test_dynamic_partition():
    """
    Test the method `EamNN._dynamic_partition`.
    """
    with tf.Graph().as_default():

        data = AlCuData()
        mode = tf.estimator.ModeKeys.TRAIN

        nn = EamAlloyNN(elements=data.elements, minimize_properties=['energy'],
                        export_properties=['energy'])

        with tf.Session() as sess:
            partitions_op, max_occurs = nn._dynamic_partition(
                data.features.descriptors, mode=mode, merge_symmetric=False)
            results = sess.run(partitions_op)

            assert_equal(len(max_occurs), 2)
            assert_equal(max_occurs['Al'], data.max_n_al)
            assert_equal(max_occurs['Cu'], data.max_n_cu)
            assert_equal(len(results), 4)
            assert_array_equal(results['AlAl'][0], data.g_al[:, [0]])
            assert_array_equal(results['AlCu'][0], data.g_al[:, [1]])
            assert_array_equal(results['CuCu'][0], data.g_cu[:, [0]])
            assert_array_equal(results['CuAl'][0], data.g_cu[:, [1]])

            assert_array_equal(results['AlAl'][1], data.m_al[:, [0]])
            assert_array_equal(results['AlCu'][1], data.m_al[:, [1]])
            assert_array_equal(results['CuCu'][1], data.m_cu[:, [0]])
            assert_array_equal(results['CuAl'][1], data.m_cu[:, [1]])

            partitions_op, _ = nn._dynamic_partition(
                data.features.descriptors, mode=mode, merge_symmetric=True)
            results = sess.run(partitions_op)

            assert_equal(len(results), 3)
            assert_array_equal(results['AlAl'][0], data.g_al[:, [0]])
            assert_array_equal(results['CuCu'][0], data.g_cu[:, [0]])

            assert_array_equal(results['AlAl'][1], data.m_al[:, [0]])
            assert_array_equal(results['CuCu'][1], data.m_cu[:, [0]])

            assert_array_equal(results['AlCu'][0],
                               np.concatenate((data.g_al[:, [1]],
                                               data.g_cu[:, [1]]), 2))
            assert_array_equal(results['AlCu'][1],
                               np.concatenate((data.m_al[:, [1]],
                                               data.m_cu[:, [1]]), 2))


@skip
def test_hidden_sizes():
    """
    Test setting hidden layer sizes of `EamAlloyNN`.
    """
    custom_potentials = {
        'AlCu': {'phi': 'zjw04'},
        'Al': {'rho': 'zjw04'},
        'CuCu': {'phi': 'zjw04'},
    }
    nn = EamAlloyNN(elements=['Al', 'Cu'],
                    hidden_sizes={'AlAl': {'phi': [128, 64]}},
                    custom_potentials=custom_potentials)
    assert_dict_equal(nn.potentials,
                      {'AlAl': {'phi': 'nn'},
                       'CuCu': {'phi': 'zjw04'},
                       'AlCu': {'phi': 'zjw04'},
                       'Al': {'rho': 'zjw04', 'embed': 'nn'},
                       'Cu': {'rho': 'nn', 'embed': 'nn'}})
    assert_equal(nn.hidden_sizes,
                 {'AlAl': {'phi': [128, 64]},
                  'CuCu': {'phi': Defaults.hidden_sizes},
                  'AlCu': {'phi': Defaults.hidden_sizes},
                  'Al': {'rho': Defaults.hidden_sizes,
                         'embed': Defaults.hidden_sizes},
                  'Cu': {'rho': Defaults.hidden_sizes,
                         'embed': Defaults.hidden_sizes}})


@skip
def test_custom_potentials():
    """
    Test setting layers of `EamAlloyNN`.
    """
    with tf.Graph().as_default():

        data = AlCuData()
        custom_potentials = {
            'AlCu': {'phi': 'zjw04'},
            'Al': {'rho': 'zjw04'},
            'CuCu': {'phi': 'zjw04'},
        }
        nn = EamAlloyNN(elements=data.elements,
                        custom_potentials=custom_potentials)
        assert_dict_equal(nn.potentials,
                          {'AlAl': {'phi': 'nn'},
                           'CuCu': {'phi': 'zjw04'},
                           'AlCu': {'phi': 'zjw04'},
                           'Al': {'rho': 'zjw04', 'embed': 'nn'},
                           'Cu': {'rho': 'nn', 'embed': 'nn'}})


@skip
def test_inference():
    """
    Test the inference of `EamAlloyNN` with mixed potentials.
    """
    with tf.Graph().as_default():

        data = AlCuData()
        batch_size = data.batch_size
        max_n_atoms = data.max_n_atoms
        mode = tf.estimator.ModeKeys.TRAIN

        nn = EamAlloyNN(elements=data.elements,
                        custom_potentials={
                            'Cu': {'rho': 'zjw04', 'embed': 'nn'},
                            'AlCu': {'phi': 'zjw04'}})

        with tf.name_scope("Inference"):
            partitions, max_occurs = nn._dynamic_partition(
                descriptors=data.features.descriptors,
                mode=mode,
                merge_symmetric=True)
            rho, _ = nn._build_rho_nn(
                descriptors=data.descriptors,
                max_occurs=max_occurs,
                mode=mode,
                verbose=True)
            embed = nn._build_embed_nn(
                rho=rho,
                max_occurs=max_occurs,
                mode=mode,
                verbose=True)
            phi, _ = nn._build_phi_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=True)
            y = tf.add(phi, embed, name='atomic')

        collection = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        assert_equal(len(collection), 41)

        collection = tf.get_collection(GraphKeys.EAM_ALLOY_NN_VARIABLES)
        assert_equal(len(collection), 25)

        collection = tf.get_collection(GraphKeys.EAM_POTENTIAL_VARIABLES)
        assert_equal(len(collection), 16)

        assert_dict_equal(max_occurs, {'Al': data.max_n_al,
                                       'Cu': data.max_n_cu})
        assert_list_equal(rho.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(embed.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(phi.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(y.shape.as_list(), [batch_size, max_n_atoms - 1])


def export_setfl_teardown():
    """
    Remove the generated setfl file.
    """
    for afile in ('AlCu.alloy.eam',
                  'Al.embed.png', 'Cu.embed.png', 'Al.rho.png', 'Cu.rho.png',
                  'AlCu.phi.png', 'AlAl.phi.png', 'CuCu.phi.png'):
        if exists(afile):
            remove(afile)


@skip
@with_setup(teardown=export_setfl_teardown)
def test_export_setfl():
    """
    Test exporting eam/alloy model of AlCuZJW04 to a setfl file.
    """
    nn = EamAlloyNN(
        elements=['Al', 'Cu'],
        custom_potentials={'Al': {'rho': 'zjw04', 'embed': 'zjw04'},
                           'Cu': {'rho': 'zjw04', 'embed': 'zjw04'},
                           'AlAl': {'phi': 'zjw04'},
                           'AlCu': {'phi': 'zjw04'},
                           'CuCu': {'phi': 'zjw04'}})
    nn.export_to_setfl('AlCu.alloy.eam',
                       nr=2000, dr=0.003, nrho=2000, drho=0.05)

    with open('AlCu.alloy.eam') as fp:
        out = []
        for i, line in enumerate(fp):
            if i < 6 or i == 4006:
                continue
            out.append(float(line.strip()))
    with open(join(test_dir(), 'lammps', 'Zhou_AlCu.alloy.eam')) as fp:
        ref = []
        for i, line in enumerate(fp):
            if i < 6 or i == 4006:
                continue
            ref.append(float(line.strip()))
    assert_array_equal(np.asarray(out), np.asarray(ref))


@skip
def test_export_setfl_from_ckpt():
    """
    Test exporting eam/alloy model by loading variables from a checkpoint.
    # TODO: to be implemented
    """
    pass


# Setup the environment for `LAMMPS`
if 'LAMMPS_COMMAND' not in os.environ:
    LAMMPS_COMMAND = '/usr/local/bin/lmp_serial'
    os.environ['LAMMPS_COMMAND'] = LAMMPS_COMMAND
else:
    LAMMPS_COMMAND = os.environ['LAMMPS_COMMAND']


def get_lammps_calculator():
    """
    Return a LAMMPS calculator for Ag.
    """
    eam_file = join(test_dir(absolute=True), 'lammps', 'Zhou_AlCu.alloy.eam')
    parameters = {'pair_style': 'eam/alloy',
                  'pair_coeff': ['* * Zhou_AlCu.alloy.eam Al Cu']}
    work_dir = join(test_dir(absolute=True), 'lammps', 'zjw04')
    if not exists(work_dir):
        os.makedirs(work_dir)

    return LAMMPS(files=[eam_file], parameters=parameters, tmp_dir=work_dir,
                  keep_tmp_files=True, keep_alive=False, no_data_file=False)


lammps = get_lammps_calculator()


def teardown():
    """
    Delete the tmp dir.
    """
    if exists(lammps.tmp_dir):
        shutil.rmtree(lammps.tmp_dir, ignore_errors=True)


@with_setup(teardown=teardown)
@skipUnless(exists(LAMMPS_COMMAND), f"{LAMMPS_COMMAND} not set!")
def test_eam_zjw04():
    """
    Test the total energy calculation of `EamAlloyNN` with `Zjw04`.
    """
    rc = 6.0

    atoms = bulk('Cu') * [2, 2, 2]
    symbols = atoms.get_chemical_symbols()
    symbols[0: 2] = ['Al', 'Al']
    atoms.set_chemical_symbols(symbols)
    elements = sorted(set(symbols))

    with tf.Graph().as_default():
        clf = EAMTransformer(rc=rc, elements=elements)
        nn = EamAlloyNN(elements=elements,
                        custom_potentials={
                            "Al": {"rho": "zjw04", "embed": "zjw04"},
                            "Cu": {"rho": "zjw04", "embed": "zjw04"},
                            "AlAl": {"phi": "zjw04"},
                            "AlCu": {"phi": "zjw04"},
                            "CuCu": {"phi": "zjw04"}})
        prediction = nn.build(
            features=clf.get_features(),
            mode=tf.estimator.ModeKeys.PREDICT,
            verbose=True)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            energy = float(sess.run(prediction.energy,
                                    feed_dict=clf.get_feed_dict(atoms)))

    atoms.calc = lammps
    assert_almost_equal(energy,
                        lammps.get_potential_energy(atoms), delta=1e-6)


if __name__ == "__main__":
    nose.run()
