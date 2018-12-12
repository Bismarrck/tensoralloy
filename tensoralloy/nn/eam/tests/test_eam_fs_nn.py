# coding=utf-8
"""
This module defines unit tests of `EamFsNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
from nose.tools import assert_list_equal, assert_dict_equal, with_setup
from nose.tools import assert_equal
from os.path import exists, join
from os import remove

from ..fs import EamFsNN
from tensoralloy.misc import skip, AttributeDict, test_dir
from tensoralloy.test_utils import assert_array_almost_equal
from tensoralloy.utils import get_elements_from_kbody_term

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
        max_n_elements = {
            'Al': data.max_n_al,
            'Fe': data.max_n_fe
        }

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
            rho, rho_values = nn._build_rho_nn(partitions, max_occurs,
                                               verbose=False)
            embed = nn._build_embed_nn(rho, max_occurs, verbose=False)

            partitions, max_occurs = nn._dynamic_partition(
                data.features, merge_symmetric=True)
            phi, phi_values = nn._build_phi_nn(partitions, max_occurs,
                                               verbose=False)
            y = tf.add(phi, embed, name='atomic')

        assert_dict_equal(max_occurs, {'Al': data.max_n_al,
                                       'Fe': data.max_n_fe})
        assert_list_equal(rho.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_equal(len(rho_values), 4)
        for kbody_term in nn.all_kbody_terms:
            center = get_elements_from_kbody_term(kbody_term)[0]
            max_n_element = max_n_elements[center]
            assert_list_equal(rho_values[kbody_term].shape.as_list(),
                              [batch_size, 1, max_n_element, data.nnl, 1])

        assert_list_equal(embed.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_list_equal(phi.shape.as_list(), [batch_size, max_n_atoms - 1])
        assert_equal(len(phi_values), 3)
        for kbody_term in nn.unique_kbody_terms:
            center, specie = get_elements_from_kbody_term(kbody_term)
            if center == specie:
                max_n_element = max_n_elements[center]
                assert_list_equal(phi_values[kbody_term].shape.as_list(),
                                  [batch_size, 1, max_n_element, data.nnl, 1])
            else:
                assert_list_equal(phi_values[kbody_term].shape.as_list(),
                                  [batch_size, 1, max_n_atoms - 1, data.nnl, 1])

        assert_list_equal(y.shape.as_list(), [batch_size, max_n_atoms - 1])


def test_export_setfl_teardown():
    """
    Remove the generated setfl file.
    """
    if exists('AlFe.fs.eam'):
        remove('AlFe.fs.eam')


@with_setup(teardown=test_export_setfl_teardown)
def test_export_setfl():
    """
    Test exporting eam/alloy model of AlCuZJW04 to a setfl file.
    """
    nn = EamFsNN(
        elements=['Al', 'Fe'],
        custom_potentials={'Al': {'embed': 'msah11'},
                           'Fe': {'embed': 'msah11'},
                           'AlAl': {'phi': 'msah11', 'rho': 'msah11'},
                           'AlFe': {'phi': 'msah11', 'rho': 'msah11'},
                           'FeFe': {'phi': 'msah11', 'rho': 'msah11'},
                           'FeAl': {'rho': 'msah11'}})

    nrho = 10000
    drho = 3.00000000000000E-2
    nr = 10000
    dr = 6.50000000000000E-4

    nn.export('AlFe.fs.eam', nr=nr, dr=dr, nrho=nrho, drho=drho,
              lattice_constants={'Al': 4.04527, 'Fe': 2.855312},
              lattice_types={'Al': 'fcc', 'Fe': 'bcc'})

    with open('AlFe.fs.eam') as fp:
        out = []
        out_key_lines = []
        for i, line in enumerate(fp):
            if i < 5:
                continue
            elif i == 5 or i == 30006:
                out_key_lines.append(line)
                continue
            out.append(float(line.strip()))
    with open(join(test_dir(), 'lammps', 'Mendelev_Al_Fe.fs.eam')) as fp:
        ref = []
        ref_key_lines = []
        for i, line in enumerate(fp):
            if i < 5:
                continue
            elif i == 5 or i == 30006:
                ref_key_lines.append(line)
                continue
            ref.append(float(line.strip()))
    assert_array_almost_equal(np.asarray(out), np.asarray(ref), delta=1e-8)
    assert_list_equal(out_key_lines, ref_key_lines)


@skip
def test_export_eam_fs_from_ckpt():
    """
    Test exporting eam/fs model by loading variables from a checkpoint.
    """
    pass


if __name__ == "__main__":
    nose.run()
