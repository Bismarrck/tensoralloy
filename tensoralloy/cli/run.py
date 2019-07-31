#!coding=utf-8
"""
Command-line programs under the `run` scope.
"""
from __future__ import print_function, absolute_import

import argparse
import os
import re
from os.path import join, exists

from tensoralloy import TrainingManager
from tensoralloy.cli.cli import CLIProgram

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class RunExperimentProgram(CLIProgram):
    """
    The program for running training experiments.
    """

    @property
    def name(self):
        return "run"

    @property
    def help(self):
        return "Run an experiment"

    @property
    def main_func(self):
        """
        Return the main function for running experiments.
        """
        def func(args: argparse.Namespace):
            manager = TrainingManager(
                args.filename)
            manager.train_and_evaluate(debug=args.debug)
            manager.export()
        return func

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            'filename',
            type=str,
            help="A cfg file to read."
        )
        subparser.add_argument(
            "--debug",
            action='store_true',
            default=False,
            help="Enabled the debugging mode.",
        )
        super(RunExperimentProgram, self).config_subparser(subparser)


class StopExperimentProgram(CLIProgram):
    """
    Stop an experiment by directly terminating the corresponding process.
    """

    @property
    def name(self):
        """
        The name of this CLI program.
        """
        return "stop"

    @property
    def help(self):
        """
        The help message.
        """
        return "Terminate a running experiment."

    def config_subparser(self, subparser: argparse.ArgumentParser):
        """
        Config the parser.
        """
        subparser.add_argument(
            "model_dir",
            type=str,
            help="The model dir of the experiment."
        )

        super(StopExperimentProgram, self).config_subparser(subparser)

    @property
    def main_func(self):
        """
        The main function.
        """
        def func(args: argparse.Namespace):
            logfile = join(args.model_dir, "logfile")
            if not exists(logfile):
                raise IOError(f"{logfile} cannot be accessed")
            pid_patt = re.compile(r".*tensorflow\s+INFO\s+pid=(\d+)")
            with open(logfile) as fp:
                for number, line in enumerate(fp):
                    m = pid_patt.search(line)
                    if m:
                        pid = m.group(1)
                        break
                    if number == 10:
                        raise IOError(f"{logfile} maybe corrupted!")
            code = os.system(f'kill -9 {pid}')
            if code == 0:
                print(f"pid={pid} killed. "
                      f"The experiment in {args.model_dir} terminated.")
            else:
                print(f"Failed to stop {args.model_dir}: error_code = {code}")
        return func
