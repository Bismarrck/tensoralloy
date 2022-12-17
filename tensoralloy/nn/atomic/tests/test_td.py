#!coding=utf-8
"""
Test the finite temperature module.
"""
import tensorflow as tf
import numpy as np
from collections import Counter
from tensoralloy.utils import ModeKeys
from tensoralloy.nn.atomic.finite_temperature import TemperatureDependentAtomicNN

    
def test_add_electron_temperature():
    etemp_ = np.ones(1, dtype=np.float64) * 0.05
    etemp_ = 0.05
    x_ = np.random.rand(1, 4, 6).astype(np.float64)

    etemp = tf.convert_to_tensor(etemp_, dtype=tf.float64, name="T")
    x = tf.convert_to_tensor(x_, dtype=tf.float64, name="x")

    y, z = TemperatureDependentAtomicNN._add_electron_temperature(
        x, etemp, "Be", ModeKeys.PREDICT, Counter())
    
    np.set_printoptions(suppress=True, linewidth=1024, precision=5)

    print(x_[0])

    with tf.Session() as sess:
        y, z = sess.run((y, z))
        print(y[0])
        print(z)


if __name__ == "__main__":
    test_add_electron_temperature()
