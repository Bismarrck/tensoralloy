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

__all__ = ["RestoreEmaVariablesHook", "LoggingTensorHook", "ProfilerHook",
           "ExamplesPerSecondHook", "WarmStartFromVariablesHook",
           "NanTensorHook"]


class WarmStartFromVariablesHook(tf.train.SessionRunHook):
    """
    This hook can be used to replace `tf_estimator.WarmStartSettings`.
    """

    def __init__(self,
                 previous_checkpoint: str,
                 ema: tf.train.ExponentialMovingAverage,
                 restore_all_variables=True,
                 restart=True):
        """
        Initialization method.

        Parameters
        ----------
        previous_checkpoint : str
            The previous checkpoint to load.
        ema : tf.train.ExponentialMovingAverage
            A function to obtain exponentially moving averaged variables.
        restore_all_variables : bool
            If True, all (global) varaibles will be restored from the ckpt. If
            False, only trainable variables and the global step variable
            (restart == False) will be restored.
        restart : bool
            A boolean. If True, the global step tensor will not be read and the
            training will start only with weights inherited from the checkpoint.
            If False, the global step tensor will also be loaded and the
            previous training will be continued.

        """
        tf.logging.info("Create WarmStartFromVariablesHook.")

        self._previous_checkpoint = previous_checkpoint
        self._ema = ema
        self._saver = None
        self._restart = restart
        self._restore_all_variables = restore_all_variables

    def begin(self):
        """
        Create restoring operations before the graph been finalized.
        """
        trainable_op_names = [var.op.name for var in tf.trainable_variables()]

        if self._ema is not None:
            var_list = self._ema.variables_to_restore()
            if self._restart and 'global_step' in var_list:
                var_list.pop('global_step')
            if not self._restore_all_variables:
                pop_list = []
                for var_op_ema_name in var_list:
                    var_op_name = var_op_ema_name.replace(
                        '/ExponentialMovingAverage', '')
                    if var_op_name not in trainable_op_names:
                        pop_list.append(var_op_ema_name)
                for var_op_name in pop_list:
                    var_list.pop(var_op_name)
            tf.logging.info('Initialize a Saver to restore EMA variables.')
        else:
            global_step = tf.train.get_global_step()
            if self._restore_all_variables:
                var_list = tf.global_variables()
                if self._restart:
                    var_list = [x for x in var_list
                                if x.op.name != global_step.op.name]
            else:
                var_list = tf.trainable_variables()
                if not self._restart:
                    var_list += [global_step, ]
            tf.logging.info('Initialize a Saver to restore variables.')

        self._saver = tf.train.Saver(var_list=var_list)

    def after_create_session(self, session, coord):
        """
        When this is called, the graph is finalized and ops can no longer be
        added to the graph.
        """
        tf.logging.info(
            f'Restore EMA variables from {self._previous_checkpoint}')
        self._saver.restore(session, self._previous_checkpoint)


class RestoreEmaVariablesHook(tf.train.SessionRunHook):
    """
    Replace parameters with their moving averages.
    This operation should be executed only once, and before any inference.

    **Note: this hook should be used for restoring parameters for evaluation. **

    References
    ----------
    https://github.com/tensorflow/tensorflow/issues/3460

    """

    def __init__(self, ema: tf.train.ExponentialMovingAverage):
        """
        Initialization method.

        Parameters
        ----------
        ema : tf.train.ExponentialMovingAverage
            A function to obtain exponentially moving averaged variables.

        """
        super(RestoreEmaVariablesHook, self).__init__()
        self._ema = ema
        self._restore_ops = None

    def begin(self):
        """
        Create restoring operations before the graph been finalized.
        """
        ema_variables = tf.moving_average_variables()
        self._restore_ops = [
            tf.assign(x, self._ema.average(x)) for x in ema_variables]

    def after_create_session(self, session, coord):
        """
        Restore the parameters right after the session been created.
        """
        session.run(self._restore_ops)


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
                'Global step should be created to use ExamplesPerSecondHook.')

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


class NanTensorHook(session_run_hook.SessionRunHook):
    """
    Monitors the loss tensor and stops training if loss is NaN.

    Can either fail with exception or just stop training.

    This is an improved version of `tf.train.NanTensorHook`.
    """

    def __init__(self, fail_on_nan_loss=True, **kwargs):
        """
        Initializes a `NanTensorHook`.

        Parameters
        ----------
        fail_on_nan_loss : bool
            A boolean, whether to raise exception when loss is NaN.

        """
        self._fail_on_nan_loss = fail_on_nan_loss
        self._global_step_tensor = None

        if len(kwargs) > 0:
            self._optional_tensors = kwargs
            assert all([isinstance(x, tf.Tensor) for x in kwargs.values()])
        else:
            self._optional_tensors = None

    def begin(self):
        """
        Called once before using the session.
        """
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use NanTensorHook.')

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """
        Called before each call to run().
        """
        args = {'global_step': self._global_step_tensor}
        if self._optional_tensors is not None:
            args.update(self._optional_tensors)
        return basic_session_run_hooks.SessionRunArgs(args)

    def after_run(self, run_context, run_values):
        """
        Called after each call to run().
        """
        step = run_values.results.pop('global_step')
        nan_tensors = [key for key, value in run_values.results.items()
                       if np.isnan(value)]
        if len(nan_tensors) > 0:
            logging.error(f"Model diverged with loss = NaN at step {step}")
            logging.error(f"NaN tensors: {', '.join(nan_tensors)}")
            if self._fail_on_nan_loss:
                raise basic_session_run_hooks.NanLossDuringTrainingError
            run_context.request_stop()
