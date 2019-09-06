#!coding=utf-8
"""
A temporarily fix to solve the confict between `MirroredStrategy` and
`ExponentialMovingAverage`.

References
----------
https://github.com/tensorflow/tensorflow/issues/27392
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import re

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.training import coordinator
from tensorflow.python.distribute import values
from tensorflow.python.ops import variable_scope
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.mirrored_strategy import _MirroredReplicaThread
from tensorflow.python.distribute.mirrored_strategy import _RequestedStop
from tensorflow.python.distribute import mirrored_strategy

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


_VARIABLE_UNIQUIFYING_REGEX = re.compile(r"_\d/")
_VARIABLE_UNIQUIFYING_REGEX_AT_END = re.compile(r"_\d$")


def _canonicalize_variable_name(name):
    # If no name is specified, uses default name "Variable".
    if name is None:
        return "Variable"
    # Replace all instances of "_<num>/" with "/"
    name = _VARIABLE_UNIQUIFYING_REGEX.sub("/", name)
    # Replace any instances of "_<num>" at the end of the string with ""
    name = _VARIABLE_UNIQUIFYING_REGEX_AT_END.sub("", name)
    return name


def make_fn(shared_variable_store, device_id):
    """Construct the variable creator function for device `device_id`.

    Constructs custom variable creator functions for the given device.
    On first device (device_id == 0), it creates the variable using the
    `next_creator`, and stores it in the provided `shared_variable_store`.
    On all other devices (device_id > 0), it tries to re-use the variable
    already created with the same name. If no such variable exists, it throws an
    error.
    Additionally, we de-uniquify variable names before checking for matches.
    This helps re-use variables which are intended to be the same but have
    different names due to variable uniquification happening upstream. Since
    this might mean we may have multiple variables with the same canonical name,
    we store them in a list per canonical name and return them in the same order
    as well.

    Args:
      shared_variable_store: A dictionary that we will use to store variables
        created on the first device, and re-used by creators for other devices.
      device_id: Integer index of the device whose creator should be
        constructed.

    Returns:
      An appropriate creator function based on device_id.

    """
    variable_scope_access_index = {}
    assert isinstance(device_id, int)

    def create_new_variable(next_creator, *args, **kwargs):
        """Create the variable using `next_creator` and store it."""
        canonical_name = _canonicalize_variable_name(kwargs.get("name"))
        v = next_creator(*args, **kwargs)

        if canonical_name not in shared_variable_store:
            shared_variable_store[canonical_name] = []
        shared_variable_store[canonical_name].append(v)
        return v

    def reuse_variable(next_creator, *args, **kwargs):
        """Re-use existing variable from store with same name (in order)."""
        del next_creator, args
        name = kwargs.get("name")
        canonical_name = _canonicalize_variable_name(name)
        replica_index = canonical_name.find('replica/')
        if replica_index != -1:
            fixed_name = canonical_name[0: replica_index] + canonical_name[(replica_index + 8):]
        else:
            fixed_name = canonical_name
        try:
            variable_index = variable_scope_access_index.get(fixed_name, 0)
            v = shared_variable_store[fixed_name][variable_index]
            # TODO(priyag): Make this variable re-use more robust
            # that the requested shape and dtype match the existing variable.
            variable_scope_access_index[fixed_name] = variable_index + 1
            return v
        except (KeyError, IndexError):
            tf.logging.error(f"fixed name: {fixed_name}")
            raise RuntimeError(f"Tried to create variable {name} with "
                               f"mismatching name on device {device_id}")

    if device_id == 0:
        return create_new_variable
    else:
        return reuse_variable


# _call_for_each_replica is not a member of MirroredStrategy so that it is
# not allowed to use anything specific to MirroredStrategy and thus
# can be shared with other distribution strategies.


# TODO(yuefengz): maybe create a common class for those who need to call this
# _call_for_each_replica.
def _call_for_each_replica(distribution, device_map, fn, args, kwargs):
    """Run `fn` in separate threads, once per replica/worker device.

    Args:
      distribution: the DistributionStrategy object.
      device_map: the DeviceMap with the devices to run `fn` on.
      fn: function to run (will be run once per replica, each in its own thread)
      args: positional arguments for `fn`
      kwargs: keyword arguments for `fn`.

    Returns:
      Merged return value of `fn` across all replicas.

    Raises:
      RuntimeError: If fn() calls get_replica_context().merge_call() a different
          number of times from the available devices.
    """
    # TODO(josh11b): Add this option once we add synchronization to variable
    # creation. Until then, this is pretty unsafe to use.
    run_concurrently = False
    if not context.executing_eagerly():
        # Needed for per-thread device, etc. contexts in graph mode.
        ops.get_default_graph().switch_to_thread_local()

    coord = coordinator.Coordinator(
        clean_stop_exception_types=(_RequestedStop, ))

    shared_variable_store = {}

    # TODO(isaprykin): Create these threads once instead of during every call.
    threads = []
    for index in range(device_map.num_replicas_in_graph):
        variable_creator_fn = make_fn(
            shared_variable_store, index)
        t = _MirroredReplicaThread(
            distribution, coord, index, device_map, variable_creator_fn, fn,
            values.select_replica(index, args),
            values.select_replica(index, kwargs))
        threads.append(t)

    for t in threads:
        t.start()

    # When `fn` starts `should_run` event is set on _MirroredReplicaThread
    # (`MRT`) threads. The execution waits until
    # `MRT.has_paused` is set, which indicates that either `fn` is
    # complete or a `get_replica_context().merge_call()` is called.  If `fn` is
    # complete, then `MRT.done` is set to True.  Otherwise, arguments
    # of `get_replica_context().merge_call` from all paused threads are grouped
    # and the `merge_fn` is performed.  Results of the
    # `get_replica_context().merge_call` are then set to `MRT.merge_result`.
    # Each such `get_replica_context().merge_call` call returns the
    # `MRT.merge_result` for that thread when `MRT.should_run` event
    # is reset again. Execution of `fn` resumes.

    try:
        with coord.stop_on_exception():
            all_done = False
            while not all_done and not coord.should_stop():
                done = []
                if run_concurrently:
                    for t in threads:
                        t.should_run.set()
                    for t in threads:
                        t.has_paused.wait()
                        t.has_paused.clear()
                        if coord.should_stop():
                            return None
                        done.append(t.done)
                else:
                    for t in threads:
                        t.should_run.set()
                        t.has_paused.wait()
                        t.has_paused.clear()
                        if coord.should_stop():
                            return None
                        done.append(t.done)
                if coord.should_stop():
                    return None
                all_done = all(done)
                if not all_done:
                    if any(done):
                        raise RuntimeError(
                            "Some replicas made a different number of "
                            "replica_context().merge_call() calls.")
                    # get_replica_context().merge_call() case
                    merge_args = values.regroup(
                        device_map, tuple(t.merge_args for t in threads))
                    merge_kwargs = values.regroup(
                        device_map, tuple(t.merge_kwargs for t in threads))
                    # We capture the name_scope of the MRT when we call merge_fn
                    # to ensure that if we have opened a name scope in the MRT,
                    # it will be respected when executing the merge function. We
                    # only capture the name_scope from the first MRT and assume
                    # it is the same for all other MRTs.
                    mtt_captured_name_scope = threads[0].captured_name_scope
                    mtt_captured_var_scope = threads[0].captured_var_scope
                    # Capture and merge the control dependencies from all the
                    # threads.
                    mtt_captured_control_deps = set()
                    for t in threads:
                        mtt_captured_control_deps.update(
                            t.captured_control_deps)
                    with ops.name_scope(mtt_captured_name_scope), \
                         ops.control_dependencies(mtt_captured_control_deps), \
                         variable_scope.variable_scope(mtt_captured_var_scope):
                        merge_result = threads[0].merge_fn(distribution,
                                                           *merge_args,
                                                           **merge_kwargs)
                    for r, t in enumerate(threads):
                        t.merge_result = values.select_replica(r, merge_result)
    finally:
        for t in threads:
            t.should_run.set()
        coord.join(threads)

    return values.regroup(device_map, tuple(t.main_result for t in threads))


class MirroredStrategy(distribute_lib.StrategyV1):
    """
    The modified `MirroredStrategy`.
    """

    __doc__ = mirrored_strategy.MirroredStrategy.__doc__

    def __init__(self, devices=None, cross_device_ops=None):
        extended = MirroredExtended(
            self, devices=devices, cross_device_ops=cross_device_ops)
        super(MirroredStrategy, self).__init__(extended)


class MirroredExtended(mirrored_strategy.MirroredExtended):
    """
    Implementation of MirroredStrategy.
    """

    def __init__(self, container_strategy, devices=None, cross_device_ops=None):
        super(MirroredExtended, self).__init__(container_strategy)
        if devices is None:
            devices = mirrored_strategy._all_devices()
        if not devices:
            raise ValueError("Got an empty `devices` list. Please make sure the "
                             "`devices` you pass in is not empty.")
        self._cross_device_ops = cross_device_ops
        self._initialize_strategy(devices)

    def _call_for_each_replica(self, fn, args, kwargs):
        """
        Use the modified `_call_for_each_replica` function.
        """
        return _call_for_each_replica(self._container_strategy(),
                                      self._device_map,
                                      fn, args, kwargs)
