#!coding=utf-8
"""
Test the Rose Equation of State Constraint.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import nose

from ase.build import bulk
from ase.units import GPa
from os.path import join
from collections import Counter
from sklearn.metrics import mean_absolute_error
from nose.tools import assert_almost_equal, assert_equal
from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.nn.constraint.rose import get_rose_constraint_loss
from tensoralloy.nn.constraint.data import built_in_crystals
from tensoralloy.nn import EamAlloyNN
from tensoralloy.nn.dataclasses import TrainParameters, OptParameters
from tensoralloy.nn.dataclasses import LossParameters, RoseLossOptions
from tensoralloy.transformer import BatchEAMTransformer
from tensoralloy.precision import precision_scope
from tensoralloy.test_utils import test_dir, assert_array_almost_equal
from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.neighbor import find_neighbor_size_of_atoms
from tensoralloy.train.dataclasses import EstimatorHyperParams
from tensoralloy.train.dataclasses import DistributeParameters


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def test_rose_eos_constraint():
    """
    Test the function `get_rose_constraint_loss`.
    """
    atoms = bulk('Mo', cubic=True)
    calc = TensorAlloyCalculator(join(test_dir(), 'models', 'Mo.zhou04.pb'))
    atoms.calc = calc
    cell = atoms.cell

    bulk_modulus = 259.0 * GPa
    e0 = calc.get_potential_energy(atoms)
    v0 = atoms.get_volume()
    alpha = np.sqrt(np.abs(9.0 * v0 * bulk_modulus / e0))
    beta = 0.5e-2
    dx = 0.10
    delta = 0.01

    y_rose_list = []
    y_list = []
    vol_list = []
    ax_list = []

    for ratio in np.linspace(1.0 - dx, 1.0 + dx, num=21, endpoint=True):
        atomsi = atoms.copy()
        atomsi.set_cell(cell * ratio, scale_atoms=True)

        # Eq. 12: x = a / a0 - 1
        x = ratio - 1
        ax = alpha * x
        inner = (1 + ax + beta * (ax**3) * (2 * x + 3) / (x + 1) ** 2)
        outer = np.exp(-ax)
        ax_list.append(ax)

        y_rose_list.append(e0 * inner * outer)
        y_list.append(calc.get_potential_energy(atomsi))
        vol_list.append(atomsi.get_volume())

    y_rose_list = np.asarray(y_rose_list)
    y_list = np.asarray(y_list)
    ax_list = np.asarray(ax_list)

    with precision_scope('high'):
        with tf.Graph().as_default():
            rc = calc.transformer.rc
            elements = ['Mo', 'Ni']
            max_occurs = Counter({'Mo': len(atoms), 'Ni': 1})
            size = find_neighbor_size_of_atoms(atoms, rc, False)
            clf = BatchEAMTransformer(rc, max_occurs, size.nij, size.nnl, 1)
            nn = EamAlloyNN(elements, 'zjw04', minimize_properties=['energy'])
            nn.attach_transformer(clf)

            protobuf = tf.convert_to_tensor(
                clf.encode(atoms).SerializeToString())
            example = clf.decode_protobuf(protobuf)

            batch = dict()
            for key, tensor in example.items():
                batch[key] = tf.expand_dims(
                    tensor, axis=0, name=tensor.op.name + '/batch')
            labels = dict(energy=batch.pop('energy'))

            train_params = TrainParameters()
            train_params.profile_steps = 0
            hparams = EstimatorHyperParams(
                train=train_params, loss=LossParameters(), opt=OptParameters(),
                distribute=DistributeParameters()
            )
            tf.train.get_or_create_global_step()

            nn.model_fn(
                features=batch,
                labels=labels,
                mode=tf_estimator.ModeKeys.TRAIN,
                params=hparams,
            )

            options = RoseLossOptions(
                crystals=[built_in_crystals['Mo']],
                beta=[beta],
                dx=dx,
                delta=delta)
            get_rose_constraint_loss(base_nn=nn, options=options)

            zjw04_vars = [var for var in tf.global_variables()
                          if var.op.name.startswith('nnEAM/Shared/Mo')
                          and 'ExponentialMovingAverage' not in var.op.name
                          and 'Adam' not in var.op.name]

            loss = tf.get_default_graph().get_tensor_by_name(
                'Rose/Mo/bcc/EOS/Loss/loss:0')
            mae = tf.get_default_graph().get_tensor_by_name(
                loss.name.replace('loss', 'mae'))

            ax_op = tf.get_default_graph().get_tensor_by_name(
                'Rose/Mo/bcc/Params/ax:0')
            alpha_op = tf.get_default_graph().get_tensor_by_name(
                'Rose/Mo/bcc/Params/alpha:0')

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                loss_val, mae_val, ax_vals, alpha_val = sess.run(
                    [loss, mae, ax_op, alpha_op])

    assert_equal(len(zjw04_vars), 20)

    eps = 1e-8
    assert_almost_equal(alpha_val, alpha, delta=eps)
    assert_array_almost_equal(ax_list, ax_vals, delta=eps)
    assert_almost_equal(mean_absolute_error(y_rose_list, y_list), mae_val,
                        delta=eps)
    assert_almost_equal(np.linalg.norm(np.subtract(y_rose_list, y_list)),
                        loss_val, delta=eps)


if __name__ == "__main__":
    nose.run()
