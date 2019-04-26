#!coding=utf-8
"""
This module defines tests of the `CLI` module.
"""
from __future__ import print_function, absolute_import

import nose
import unittest

from os.path import join, exists
from os import remove
from nose.tools import assert_equal

from tensoralloy.cli import PrintEvaluationSummaryProgram
from tensoralloy.test_utils import test_dir

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class PrintSummaryTest(unittest.TestCase):

    def setUp(self):
        """
        The setup function.
        """
        self.csv_file = join(test_dir(), 'summary.csv')

    def test_print_summary(self):
        """
        Test `PrintEvaluationSummaryProgram`.
        """
        program = PrintEvaluationSummaryProgram()
        df = program.print_evaluation_summary(join(test_dir(), "logfile"))
        assert_equal(df.index.tolist(), [200, 400])
        assert_equal(exists(self.csv_file), True)

    def tearDown(self):
        """
        The cleanup function.
        """
        if exists(self.csv_file):
            remove(self.csv_file)


if __name__ == "__main__":
    nose.run()
