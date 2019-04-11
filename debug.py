from __future__ import print_function, absolute_import

from tensorflow_estimator import estimator as tf_estimator

from tensoralloy.nn import EamAlloyNN
from tensoralloy.transformer import EAMTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


clf = EAMTransformer(6.5, ['Mo', 'Ni'])
nn = EamAlloyNN(['Mo', 'Ni'], 'zjw04')
nn.attach_transformer(clf)
nn.build(clf.get_placeholder_features(), mode=tf_estimator.ModeKeys.PREDICT)
