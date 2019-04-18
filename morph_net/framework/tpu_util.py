"""Utility functions for handling TPU graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_variable_name(read_variable_op):
  assert read_variable_op.type == 'ReadVariableOp'
  op = read_variable_op
  while op.type != 'VarHandleOp':
    assert len(op.inputs) == 1
    op = op.inputs[0].op
  return op.name


def maybe_convert_to_variable(tensor):
  """Convert TPU variable to be usable outside a while loop.

  Args:
    tensor: A tf.Tensor.

  Returns:
    If tensor is the output of reading a ResourceVariable, replace it with an
    equivalent tensor produced outside the while loop. Otherwise, return the
    tensor unmodified.
  """
  op = tensor.op
  if op.type != 'ReadVariableOp':
    # No need to convert.
    return tensor
  with tf.variable_scope(
      # The name already contains (as slashes) all the scope we need.
      name_or_scope='',
      # We intend toobtaining a reference to an existing variable.
      reuse=True,
  ):
    return tf.get_variable(get_variable_name(op))
