"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import tensorflow as tf
from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def getConcatVariable(name, shape, dtype, numShards):
  """Get a sharded variable concatenated into one tensor."""
  shardedVariable = getShardedVariable(name, shape, dtype, numShards)
  if len(shardedVariable) == 1:
    return shardedVariable[0]

  concatName = name + "/concat"
  concatFullName = vs.get_variable_scope().name + "/" + concatName + ":0"
  for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
    if value.name == concatFullName:
      return value

  concatVariable = array_ops.concat(shardedVariable, 0, name=concatName)
  ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
                        concatVariable)
  return concatVariable


def getShardedVariable(name, shape, dtype, numShards):
  """Get a list of sharded variables with the given dtype."""
  if numShards > shape[0]:
    raise ValueError("Too many shards: shape=%s, numShards=%d" %
                     (shape, numShards))
  unitShardSize = int(math.floor(shape[0] / numShards))
  remainingRows = shape[0] - unitShardSize * numShards

  shards = []
  for i in range(numShards):
    currentSize = unitShardSize
    if i < remainingRows:
      currentSize += 1
    shards.append(vs.get_variable(name + "_%d" % i, [currentSize] + shape[1:],
                                  dtype=dtype))
  return shards


class CoupledInputForgetGateLSTMCell(rnn_cell_impl.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  The coupling of input and forget gate is based on:

    http://arxiv.org/pdf/1503.04069.pdf

  Greff et al. "LSTM: A Search Space Odyssey"

  The class uses optional peep-hole connections, and an optional projection
  layer.
  """

  def __init__(self, numUnits, usePeepholes=False,
               initializer=None, numProj=None, projClip=None,
               numUnitShards=1, numProShards=1,
               forgetBias=1.0, stateIsTuple=True,
               activation=math_ops.tanh, reuse=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      numUnits: int, The number of units in the LSTM cell
      usePeepholes: bool, set True to enable diagonal/peephole connections.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      numProj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      projClip: (optional) A float value.  If `numProj > 0` and `projClip` is
      provided, then the projected values are clipped elementwise to within
      `[-projClip, projClip]`.
      numUnitShards: How to split the weight matrix.  If >1, the weight
        matrix is stored across numUnitShards.
      numProShards: How to split the projection matrix.  If >1, the
        projection matrix is stored across numProShards.
      forgetBias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      stateIsTuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      activation: Activation function of the inner states.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(CoupledInputForgetGateLSTMCell, self).__init__(_reuse=reuse)
    if not stateIsTuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use stateIsTuple=True.", self)
    self._numUnits = numUnits
    self._usePeepholes = usePeepholes
    self._initializer = initializer
    self._numProj = numProj
    self._projClip = projClip
    self._numUnitShards = numUnitShards
    self._numProShards = numProShards
    self._forgetBias = forgetBias
    self._stateIsTuple = stateIsTuple
    self._activation = activation
    self._reuse = reuse

    if numProj:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(numUnits, numProj)
                          if stateIsTuple else numUnits + numProj)
      self._output_size = numProj
    else:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(numUnits, numUnits)
                          if stateIsTuple else 2 * numUnits)
      self._output_size = numUnits

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x numUnits.
      state: if `stateIsTuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `stateIsTuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
      scope: VariableScope for the created subgraph; defaults to "LSTMCell".

    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           numProj if numProj was set,
           numUnits otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    sigmoid = math_ops.sigmoid

    numProj = self._numUnits if self._numProj is None else self._numProj

    if self._stateIsTuple:
      (cPrev, mPrev) = state
    else:
      cPrev = array_ops.slice(state, [0, 0], [-1, self._numUnits])
      mPrev = array_ops.slice(state, [0, self._numUnits], [-1, numProj])

    dtype = inputs.dtype
    inputSize = inputs.get_shape().with_rank(2)[1]

    if inputSize.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # Input gate weights
    self.w_xi = tf.get_variable("_w_xi", [inputSize.value, self._numUnits])
    self.w_hi = tf.get_variable("_w_hi", [self._numUnits, self._numUnits])
    self.w_ci = tf.get_variable("_w_ci", [self._numUnits, self._numUnits])
    # Output gate weights
    self.w_xo = tf.get_variable("_w_xo", [inputSize.value, self._numUnits])
    self.w_ho = tf.get_variable("_w_ho", [self._numUnits, self._numUnits])
    self.w_co = tf.get_variable("_w_co", [self._numUnits, self._numUnits])

    # Cell weights
    self.w_xc = tf.get_variable("_w_xc", [inputSize.value, self._numUnits])
    self.w_hc = tf.get_variable("_w_hc", [self._numUnits, self._numUnits])

    # Initialize the bias vectors
    self.b_i = tf.get_variable("_b_i", [self._numUnits], initializer=init_ops.zeros_initializer())
    self.b_c = tf.get_variable("_b_c", [self._numUnits], initializer=init_ops.zeros_initializer())
    self.b_o = tf.get_variable("_b_o", [self._numUnits], initializer=init_ops.zeros_initializer())

    i_t = sigmoid(math_ops.matmul(inputs, self.w_xi) +
                  math_ops.matmul(mPrev, self.w_hi) +
                  math_ops.matmul(cPrev, self.w_ci) +
                  self.b_i)
    c_t = ((1 - i_t) * cPrev + i_t * self._activation(math_ops.matmul(inputs, self.w_xc) +
                                                       math_ops.matmul(mPrev, self.w_hc) + self.b_c))

    o_t = sigmoid(math_ops.matmul(inputs, self.w_xo) +
                  math_ops.matmul(mPrev, self.w_ho) +
                  math_ops.matmul(c_t, self.w_co) +
                  self.b_o)

    h_t = o_t * self._activation(c_t)

    newState = (rnn_cell_impl.LSTMStateTuple(c_t, h_t) if self._stateIsTuple else
                 array_ops.concat([c_t, h_t], 1))
    return h_t, newState