# coding=utf-8
"""
This module defines the command-line main function of `tensoralloy`.
"""
from __future__ import print_function, absolute_import

import pandas as pd
import numpy as np
import argparse
import re

from os.path import exists, dirname, join

from tensoralloy.cli.build import BuildDatabaseProgram
from tensoralloy.cli.cli import CLIProgram
from tensoralloy.cli.compute import ComputeMetricsProgram
from tensoralloy.cli.export import ExportModelProgram
from tensoralloy.cli.run import RunExperimentProgram, StopExperimentProgram

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class PrintEvaluationSummaryProgram(CLIProgram):
    """
    Print the summary of all evaluations of an experiment.
    """

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "print"

    @property
    def help(self):
        """
        The help message.
        """
        return "Print the summary of the evaluation results."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'logfile',
            type=str,
            help="The logfile of an experiment."
        )

        super(PrintEvaluationSummaryProgram, self).config_subparser(subparser)

    @staticmethod
    def print_evaluation_summary(logfile) -> pd.DataFrame:
        """
        Summarize the evalutaion results of the logfile.
        """

        global_step_patt = re.compile(r".*tensorflow\s+INFO\s+Saving\sdict"
                                      r"\sfor\sglobal\sstep\s(\d+):(.*)")
        key_value_pair_patt = re.compile(r"\s+(.*)\s=\s([0-9.-]+)")
        pid_patt = re.compile(r".*tensorflow\s+INFO\s+pid=(\d+)")
        results = {}

        with open(logfile) as fp:
            for line in fp:
                line = line.strip()
                if pid_patt.search(line):
                    results.clear()
                    continue

                m = global_step_patt.search(line)
                if not m:
                    continue

                for s in m.group(2).split(','):
                    key_value_pair = key_value_pair_patt.search(s)
                    key = key_value_pair.group(1)
                    val = key_value_pair.group(2)

                    if key == 'global_step':
                        convert_fn = int

                    elif key.startswith('Elastic'):
                        def convert_fn(_x):
                            """ Convert the string to int. """
                            return "%.1f" % np.round(float(_x), 1)

                        if 'Constraints' in key:
                            key = key[8:].replace('/Constraints', '')
                        else:
                            key = key[8:].replace('/Cijkl', '')
                    else:
                        convert_fn = float

                    results[key] = results.get(key, []) + [convert_fn(val)]

        df = pd.DataFrame(results)
        df.set_index('global_step', inplace=True)

        print(df.to_string())

        with open(join(dirname(logfile), 'summary.csv'), 'w') as fp:
            fp.write(df.to_csv())

        return df

    @property
    def main_func(self):
        """
        The main function of this program.
        """
        def func(args: argparse.Namespace):
            logfile = args.logfile
            if not exists(logfile):
                raise IOError(f"The logfile {logfile} cannot be accessed!")
            self.print_evaluation_summary(logfile)
        return func


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar='Build a database or run an experiment.'
    )

    for prog in (BuildDatabaseProgram(),
                 RunExperimentProgram(),
                 ExportModelProgram(),
                 StopExperimentProgram(),
                 PrintEvaluationSummaryProgram(),
                 ComputeMetricsProgram()):
        subparser = subparsers.add_parser(prog.name, help=prog.help)
        prog.config_subparser(subparser)

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
    else:
        args.func(args)
