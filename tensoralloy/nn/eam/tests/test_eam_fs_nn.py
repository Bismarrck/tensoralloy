# coding=utf-8
"""
This module defines unit tests of `EamFsNN`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose
import os
import shutil

from tensorflow_estimator import estimator as tf_estimator
from unittest import skipUnless, skip
from nose.tools import assert_list_equal, assert_dict_equal, with_setup
from nose.tools import assert_equal, assert_almost_equal
from os.path import exists, join
from os import remove
from ase.calculators.lammpsrun import LAMMPS
from ase.build import bulk

from tensoralloy.nn.eam import EamFsNN
from tensoralloy.transformer import EAMTransformer
from tensoralloy.test_utils import assert_array_almost_equal, test_dir
from tensoralloy.utils import get_elements_from_kbody_term, GraphKeys, AttributeDict

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class AlFeFakeData:
    """
    A fake dataset for unit tests in this module.
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

        shape_al = (4,
                    self.batch_size,
                    self.max_n_terms,
                    self.max_n_al,
                    self.nnl)
        shape_fe = (4,
                    self.batch_size,
                    self.max_n_terms,
                    self.max_n_fe,
                    self.nnl)

        self.g_al = np.random.randn(*shape_al)
        self.m_al = np.random.randint(0, 2, shape_al[1:]).astype(np.float64)
        self.g_fe = np.random.randn(*shape_fe)
        self.m_fe = np.random.randint(0, 2, shape_fe[1:]).astype(np.float64)

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

        data = AlFeFakeData()
        mode = tf_estimator.ModeKeys.EVAL
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
                descriptors=data.features.descriptors,
                mode=mode,
                merge_symmetric=False)
            rho, rho_values = nn._build_rho_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=False)

            embed = nn._build_embed_nn(
                rho=rho,
                max_occurs=max_occurs,
                mode=mode,
                verbose=False)

            partitions, max_occurs = nn._dynamic_partition(
                descriptors=data.features.descriptors,
                mode=mode,
                merge_symmetric=True)
            phi, phi_values = nn._build_phi_nn(
                partitions=partitions,
                max_occurs=max_occurs,
                mode=mode,
                verbose=False)
            y = tf.add(phi, embed, name='atomic')

        collection = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        assert_equal(len(collection), 25)

        collection = tf.get_collection(GraphKeys.EAM_FS_NN_VARIABLES)
        assert_equal(len(collection), 25)

        collection = tf.get_collection(GraphKeys.EAM_POTENTIAL_VARIABLES)
        assert_equal(len(collection), 0)

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


def export_setfl_teardown():
    """
    Remove the generated setfl file.
    """
    for obj in ('AlFe.fs.eam',
                'Al.embed.png', 'Fe.embed.png',
                'AlAl.rho.png', 'AlFe.rho.png', 'FeFe.rho.png', 'FeAl.rho.png',
                'AlFe.phi.png', 'AlAl.phi.png', 'FeFe.phi.png'):
        if exists(obj):
            remove(obj)


@with_setup(teardown=export_setfl_teardown)
def test_export_setfl():
    """
    Test exporting eam/alloy model of AlCuZJW04 to a setfl file.
    """
    nn = EamFsNN(
        elements=['Al', 'Fe'],
        custom_potentials="msah11")

    nrho = 10000
    drho = 3.00000000000000E-2
    nr = 10000
    dr = 6.50000000000000E-4

    nn.export_to_setfl('AlFe.fs.eam', nr=nr, dr=dr, nrho=nrho, drho=drho,
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
    eam_file = join(test_dir(absolute=True), 'lammps', 'Mendelev_Al_Fe.fs.eam')
    parameters = {'pair_style': 'eam/fs',
                  'pair_coeff': ['* * Mendelev_Al_Fe.fs.eam Al Fe']}
    work_dir = join(test_dir(absolute=True), 'lammps', 'msah11')
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
def test_eam_fs_msah11():
    """
    Test the total energy calculation of `EamFsNN` with `Msah11`.
    """
    rc = 6.5

    atoms = bulk('Fe') * [2, 2, 2]
    symbols = atoms.get_chemical_symbols()
    symbols[0: 2] = ['Al', 'Al']
    atoms.set_chemical_symbols(symbols)
    elements = sorted(set(symbols))

    with tf.Graph().as_default():
        clf = EAMTransformer(rc=rc, elements=elements)
        nn = EamFsNN(elements=elements,
                     export_properties=['energy', 'forces', 'stress'],
                     custom_potentials={
                         "Al": {"embed": "msah11"},
                         "Fe": {"embed": "msah11"},
                         "AlAl": {"phi": "msah11", "rho": "msah11"},
                         "AlFe": {"phi": "msah11", "rho": "msah11"},
                         "FeFe": {"phi": "msah11", "rho": "msah11"},
                         "FeAl": {"phi": "msah11", "rho": "msah11"}})
        nn.attach_transformer(clf)
        predictions = nn.build(
            features=clf.get_placeholder_features(),
            mode=tf_estimator.ModeKeys.PREDICT,
            verbose=True)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            result = sess.run(predictions, feed_dict=clf.get_feed_dict(atoms))

    atoms.calc = lammps
    lammps.calculate(atoms)

    assert_almost_equal(result['energy'],
                        lammps.get_potential_energy(atoms), delta=1e-6)
    assert_array_almost_equal(result['forces'],
                              lammps.get_forces(atoms), delta=1e-9)


if __name__ == "__main__":
    nose.run()
