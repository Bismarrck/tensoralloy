# coding=utf-8
"""
This module defines unit tests of `EamNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_equal, assert_tuple_equal
from nose.tools import assert_dict_equal, assert_list_equal, with_setup
from os.path import join, exists
from os import remove

from ..alloy import EamAlloyNN
from tensoralloy.misc import AttributeDict, test_dir, skip, Defaults
from tensoralloy.test_utils import assert_array_equal

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Data:
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


def test_dynamic_stitch():
    """
    Test the method `EamNN._dynamic_stitch`.
    """
    with tf.Graph().as_default():

        data = Data()

        nn = EamAlloyNN(elements=data.elements, forces=False)

        op = nn._dynamic_stitch(data.atomic_splits, symmetric=False)
        symm_op = nn._dynamic_stitch(data.symmetric_atomic_splits,
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


def test_dynamic_partition():
    """
    Test the method `EamNN._dynamic_partition`.
    """
    with tf.Graph().as_default():

        data = Data()

        nn = EamAlloyNN(elements=data.elements, forces=False)

        with tf.Session() as sess:
            partitions_op, max_occurs = nn._dynamic_partition(
                data.features, merge_symmetric=False)
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
                data.features, merge_symmetric=True)
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


def test_custom_potentials():
    """
    Test setting layers of `EamAlloyNN`.
    """
    with tf.Graph().as_default():

        data = Data()
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


def test_inference():
    """
    Test the inference of `EamAlloyNN` with mixed potentials.
    """
    with tf.Graph().as_default():

        data = Data()
        batch_size = data.batch_size
        max_n_atoms = data.max_n_atoms

        nn = EamAlloyNN(elements=data.elements,
                        custom_potentials={
                            'Cu': {'rho': 'zjw04', 'embed': 'nn'},
                            'AlCu': {'phi': 'zjw04'}})

        with tf.name_scope("Inference"):
            partitions, max_occurs = nn._dynamic_partition(
                data.features, merge_symmetric=True)
            rho, _ = nn._build_rho_nn(data.descriptors, verbose=True)
            embed = nn._build_embed_nn(rho, max_occurs, verbose=True)
            phi, _ = nn._build_phi_nn(partitions, verbose=True)
            y = tf.add(phi, embed, name='atomic')

        assert_dict_equal(max_occurs, {'Al': data.max_n_al,
                                       'Cu': data.max_n_cu})
        assert_list_equal(rho.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(embed.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(phi.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(y.shape.as_list(), [batch_size, max_n_atoms - 1])


def test_export_setfl_teardown():
    """
    Remove the generated setfl file.
    """
    if exists('AlCu.alloy.eam'):
        remove('AlCu.alloy.eam')


@with_setup(teardown=test_export_setfl_teardown)
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
    nn.export('AlCu.alloy.eam', nr=2000, dr=0.003, nrho=2000, drho=0.05)

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


if __name__ == "__main__":
    nose.run()
