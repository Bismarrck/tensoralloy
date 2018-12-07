# coding=utf-8
"""
This module defines unit tests of `EamFsNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_list_equal, assert_dict_equal

from ..fs import EamFsNN
from tensoralloy.misc import skip, AttributeDict

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
        self.max_n_fe = 7
        self.max_n_atoms = self.max_n_al + self.max_n_fe + 1
        self.nnl = 10
        self.elements = sorted(['Fe', 'Al'])

        shape_al = (self.batch_size, self.max_n_terms, self.max_n_al, self.nnl)
        shape_fe = (self.batch_size, self.max_n_terms, self.max_n_fe, self.nnl)

        self.g_al = np.random.randn(*shape_al)
        self.m_al = np.random.randint(0, 2, shape_al).astype(np.float64)
        self.g_fe = np.random.randn(*shape_fe)
        self.m_fe = np.random.randint(0, 2, shape_fe).astype(np.float64)

        with tf.name_scope("Inputs"):
            self.descriptors = AttributeDict(
                Al=(tf.convert_to_tensor(self.g_al, tf.float64, 'g_al'),
                    tf.convert_to_tensor(self.m_al, tf.float64, 'm_al')),
                Fe=(tf.convert_to_tensor(self.g_fe, tf.float64, 'g_fe'),
                    tf.convert_to_tensor(self.m_fe, tf.float64, 'm_fe')))
            self.positions = tf.constant(
                np.random.rand(1, self.max_n_atoms, 3),
                dtype=tf.float64,
                name='positions')
            self.mask = np.ones((self.batch_size, self.max_n_atoms), np.float64)
            self.features = AttributeDict(
                descriptors=self.descriptors, positions=self.positions,
                mask=self.mask)


def test_inference():
    """
    Test the inference of `EamFsNN` with mixed potentials.
    """
    with tf.Graph().as_default():

        data = Data()
        batch_size = data.batch_size
        max_n_atoms = data.max_n_atoms

        nn = EamFsNN(elements=['Al', 'Fe'],
                     custom_potentials={'Al': {'embed': 'msah11'},
                                        'Fe': {'embed': 'nn'},
                                        'AlAl': {'rho': 'msah11', 'phi': 'nn'},
                                        'AlFe': {'rho': 'nn', 'phi': 'nn'},
                                        'FeFe': {'rho': 'msah11', 'phi': 'nn'},
                                        'FeAl': {'rho': 'msah11'}})

        with tf.name_scope("Inference"):
            partitions, max_occurs = nn._dynamic_partition(
                data.features, merge_symmetric=False)
            rho, _ = nn._build_rho_nn(partitions, verbose=False)
            embed = nn._build_embed_nn(rho, max_occurs, verbose=False)

            partitions, max_occurs = nn._dynamic_partition(
                data.features, merge_symmetric=True)
            phi, _ = nn._build_phi_nn(partitions, verbose=False)
            y = tf.add(phi, embed, name='atomic')

        assert_dict_equal(max_occurs, {'Al': data.max_n_al,
                                       'Fe': data.max_n_fe})
        assert_list_equal(rho.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(embed.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(phi.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(y.shape.as_list(), [batch_size, max_n_atoms - 1])


def test_export_eam_fs_file():
    pass


@skip
def test_export_eam_fs_from_ckpt():
    """
    Test exporting eam/fs model by loading variables from a checkpoint.
    """
    pass


if __name__ == "__main__":
    nose.run()
