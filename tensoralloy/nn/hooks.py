# coding=utf-8
"""
This module defines custom session hooks for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import shutil
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import training_util

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class ProfilerHook(basic_session_run_hooks.ProfilerHook):
    """
    A modified implementation of `ProfilerHook`.
    """

    def __init__(self,
                 save_steps=None,
                 save_secs=None,
                 output_dir="",
                 show_dataflow=True,
                 show_memory=False):
        """
        Initialization method.
        """
        super(ProfilerHook, self).__init__(
            save_steps=save_steps, save_secs=save_secs, output_dir=output_dir,
            show_dataflow=show_dataflow, show_memory=show_memory)

        if tf.gfile.Exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        tf.gfile.MakeDirs(output_dir)


class LoggingTensorHook(basic_session_run_hooks.LoggingTensorHook):
    """
    A modified implementation of `LoggingTensorHook`.
    """

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(precision=6, suppress=True, linewidth=75,
                            floatmode='fixed')
        elapsed_secs, _ = self._timer.update_last_triggered_step(
            self._iter_count)
        if self._formatter:
            logging.info(self._formatter(tensor_values))
        else:
            stats = []
            line = []
            length = 0
            for tag in self._tag_order:
                out = "%s = %s" % (tag, np.array2string(tensor_values[tag]))
                length += len(out)
                if length > 120:
                    stats.append(', '.join(line))
                    length = 0
                    line = []
                line.append(out)
            stats.append(', '.join(line))
            contents = "\n".join(stats)
            if elapsed_secs is not None:
                logging.info('({:.3f} sec)\n'.format(elapsed_secs) + contents)
            else:
                logging.info('\n' + contents)
        np.set_printoptions(**original)


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """
    Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.

    """

    def __init__(self, batch_size, every_n_steps=100, every_n_secs=None):
        """
        Initializer for ExamplesPerSecondHook.

        Parameters
        ----------
        batch_size : int
            Total batch size used to calculate examples/second from global time.
        every_n_steps : int
            Log stats every n steps.
        every_n_secs : int
            Log stats every n seconds.

        """
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)
        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size
        self._global_step_tensor = None

    def begin(self):
        """
        Called once before using the session.
        """
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.')

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """
        Called before each call to run().
        """
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context: tf.train.SessionRunContext, run_values):
        """
        Called after each call to run().
        """
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = \
                self._timer.update_last_triggered_step(global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                steps_per_min = steps_per_sec * 60.0
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                        self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                # Average examples/sec followed by current examples/sec
                tf.logging.info(
                    '%s: %6.1f (%6.1f), step = %7d, %s: %8.1f',
                    'Average examples/sec', average_examples_per_sec,
                    current_examples_per_sec, self._total_steps,
                    'steps/minute', steps_per_min)
