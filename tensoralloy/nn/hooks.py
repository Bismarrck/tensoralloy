# coding=utf-8
"""
This module defines custom session hooks for this project.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import shutil

from tensorflow_estimator import estimator as tf_estimator
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import training_util

from tensoralloy.utils import ModeKeys
from tensoralloy.nn.dataclasses import CkptParameters


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

__all__ = ["RestoreEmaVariablesHook", "LoggingTensorHook", "ProfilerHook",
           "ExamplesPerSecondHook", "WarmStartFromVariablesHook",
           "NanTensorHook"]


class WarmStartFromVariablesHook(tf_estimator.SessionRunHook):
    """
    This hook can be used to replace `tf_estimator.WarmStartSettings`.
    """

    def __init__(self,
                 ckpt_params: CkptParameters,
                 ema: tf.train.ExponentialMovingAverage,
                 reset_global_step=False):
        """
        Initialization method.

        Parameters
        ----------
        ckpt_params : str
            The checkpoint related options.
        ema : tf.train.ExponentialMovingAverage
            A function to obtain exponentially moving averaged variables.
        reset_global_step : bool
            A boolean. If True, the global step tensor will not be read and the
            training will start only with weights inherited from the checkpoint.
            If False, the global step tensor will also be loaded and the
            previous training will be continued.

        """
        tf.logging.info("Create WarmStartFromVariablesHook.")

        self._ckpt_params = ckpt_params
        self._ema = ema
        self._reset_global_step = reset_global_step

    @staticmethod
    def is_variable_for_optimizer(var_op_name: str):
        """
        Is this variable created by an optimizer?
        """
        if "Adam" in var_op_name:
            return True
        return False

    def begin(self):
        """
        Create restoring operations before the graph been finalized.
        """
        assignment_map = {}

        if self._ckpt_params.use_ema_variables:
            for name, var in self._ema.variables_to_restore().items():
                ema_var = self._ema.average(var)
                if ema_var is None:
                    if self._ckpt_params.restore_all_variables:
                        assignment_map[name] = var
                else:
                    assignment_map[name] = self._ema.average(var)
                    assignment_map[var.op.name] = var
        else:
            if self._ckpt_params.restore_all_variables:
                var_list = tf.global_variables()
            else:
                var_list = tf.trainable_variables()
            for var in var_list:
                if "MovingAverage" not in var.op.name:
                    assignment_map[var.op.name] = var

        if self._reset_global_step:
            if "global_step" in assignment_map:
                del assignment_map["global_step"]

        del_keys = []
        if not self._ckpt_params.restore_optimizer_vars:
            for key in assignment_map:
                if self.is_variable_for_optimizer(key):
                    del_keys.append(key)
        for key in del_keys:
            del assignment_map[key]

        tf.train.init_from_checkpoint(self._ckpt_params.checkpoint_filename,
                                      assignment_map=assignment_map)


class RestoreEmaVariablesHook(tf_estimator.SessionRunHook):
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

    def __init__(self, tensors, mode: ModeKeys, every_n_iter=None,
                 every_n_secs=None, at_end=False, formatter=None):
        """
        Initialization method.
        """
        super(LoggingTensorHook, self).__init__(
            tensors, every_n_iter=every_n_iter, every_n_secs=every_n_secs,
            at_end=at_end, formatter=formatter)
        self._mode = mode
        self._iter_count = 0
        self._should_trigger = False

    def after_create_session(self, session, coord):
        """
        When this is called, the graph is finalized and ops can no longer be
        added to the graph.
        """
        if self._mode != ModeKeys.TRAIN:
            return

        tf.logging.info("All Trainable Ops: ")
        for i, var in enumerate(tf.trainable_variables()):
            tf.logging.info(f"{i:3d}. {var.op.name:s}")

        if isinstance(self._tensors, dict):
            ops = self._tensors.values()
        else:
            ops = self._tensors

        tf.logging.info("All monitored Ops: ")
        for i, var in enumerate(ops):
            tf.logging.info(f"{i:3d}. {var.op.name:s}")

        self._iter_count = session.run(tf.train.get_or_create_global_step())
        if self._iter_count > 0:
            self._should_trigger = True
        tf.logging.info(f"Global step starts from: {self._iter_count}")

    def begin(self):
        """ Called once before using the session. """
        super(LoggingTensorHook, self).begin()

    def before_run(self, run_context):
        """ Called before each call to run(). """
        return super(LoggingTensorHook, self).before_run(run_context)

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

    def after_run(self, run_context, run_values):
        """ Called after each call to run(). """
        super(LoggingTensorHook, self).after_run(run_context, run_values)

    def end(self, session):
        """ Called at the end of session. """
        super(LoggingTensorHook, self).end(session)


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """
    Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.

    """

    def __init__(self, batch_size_per_replica, num_replicas, every_n_steps=100,
                 every_n_secs=None):
        """
        Initializer for ExamplesPerSecondHook.

        Parameters
        ----------
        batch_size_per_replica : int
            The batch size for each replica.
        num_replicas : int
            The number of parallel replicas.
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
        self._batch_size_per_replica = batch_size_per_replica
        self._num_replicas = num_replicas
        self._global_batch_size = batch_size_per_replica * num_replicas
        self._global_step_tensor = None

    def begin(self):
        """
        Called once before using the session.
        """
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use ExamplesPerSecondHook.')

    def after_create_session(self, session, coord):
        """
        When this is called, the graph is finalized and ops can no longer be
        added to the graph.
        """
        self._total_steps = session.run(tf.train.get_or_create_global_step())

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """
        Called before each call to run().
        """
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self,
                  run_context: tf_estimator.SessionRunContext,
                  run_values):
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

                average_examples_per_sec = self._global_batch_size * (
                        self._total_steps / self._step_train_time)
                examples_per_sec = steps_per_sec * self._global_batch_size
                # Average examples/sec followed by current examples/sec
                tf.logging.info(
                    f'Average examples/sec: {average_examples_per_sec:6.1f} '
                    f'({examples_per_sec:6.1f}), step = {self._total_steps:7d},'
                    f' steps/minute: {steps_per_min:8.1f}')


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
