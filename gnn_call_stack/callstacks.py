from __future__ import annotations

import abc
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from clrs._src.nets import _MessagePassingScanState


def network_from_string(definition: str) -> Tuple[hk.Sequential, int]:
  comps = definition.split("_")
  last_dim = -1
  layers = []
  # TODO make sure this is actually what they are using
  # initializer = jnp.zeros
  for c in comps:
    if c.isdigit():
      last_dim = int(c)
      initializer = hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(last_dim))
      layers.append(hk.Linear(last_dim, w_init=initializer, b_init=jnp.zeros))
    else:
      layers.append(getattr(jax.nn, c))
  return hk.Sequential(layers), last_dim

class CallstackModule(abc.ABC, hk.Module):
  """
  IMPORTANT: All names must end in "CallstackModule". Not only, because this part is cut off when getting the module
  from the commandline parameter but more importantly because parameters containing "callstack_module" will not be
  filtered out in the parameter update. If any network used in here doesn't learn, this is a likely cause.
  """

  def __init__(self, **kwargs):
    super().__init__()
    pass

  @abc.abstractmethod
  def initial_values(self, batch_size: int, num_steps: int, num_nodes: int):
    pass
  @abc.abstractmethod
  def get_top(self, mp_state: _MessagePassingScanState):
    pass

  @abc.abstractmethod
  def __call__(self, stack, stack_pointers, hint_preds, hiddens, graph_fts):
    pass

  @abc.abstractmethod
  def add_to_features(self, top_stack, node_fts, edge_fts, graph_fts):
    pass

class NoneCallstackModule(CallstackModule):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def initial_values(self, batch_size: int, num_steps: int, num_nodes: int):
    return None, None
  def get_top(self, mp_state: _MessagePassingScanState):
    return None
  def __call__(self, stack, stack_pointers, hint_preds, hiddens, graph_fts):
    return None, None
  def add_to_features(self, top_stack, node_fts, edge_fts, graph_fts):
    """
    :param top_stack: as returned by get_top()
    :param node_fts: [batch_size, num_nodes, hidden_dim]
    :param edge_fts: [batch_size, num_nodes, num_nodes, hidden_dim]
    :param graph_fts: [batch_size, hidden_dim]
    :returns: modified (node_fts, edge_fts, graph_fts)
    """
    return node_fts, edge_fts, graph_fts


class GraphLevelCallstackModule(CallstackModule):

  def __init__(self, num_hiddens_for_stack: int, stack_pooling_fun: str, value_network: str, sum_fts : bool, **kwargs):
    super().__init__(**kwargs)
    self.num_hiddens_for_stack = num_hiddens_for_stack
    if hasattr(jnp, stack_pooling_fun):
      self.stack_pooling_fun = getattr(jnp, stack_pooling_fun)
    else:
      self.stack_pooling_fun = None
      self.att_net, out_dim = network_from_string(stack_pooling_fun)
      if out_dim != 1:
        raise ValueError(f"Output dimension of attention network {stack_pooling_fun} is {out_dim}, not 1.")
    if value_network is None or value_network == "":
      self.value_network = None
    else:
      self.value_network, output_dim = network_from_string(value_network)
      if output_dim != num_hiddens_for_stack:
        raise ValueError(
          f"Output dimension of the mlp ({output_dim}) must be equal to the stack dimension {num_hiddens_for_stack}!")
    self.sum_fts = sum_fts
    # self.print_counter = 0

  def initial_values(self, batch_size: int, num_steps: int, num_nodes: int):
    stack = jnp.zeros((batch_size, num_steps + 1, self.num_hiddens_for_stack))
    stack_pointers = jnp.zeros((batch_size,), dtype=int)
    return stack, stack_pointers
  def get_top(self, mp_state: _MessagePassingScanState):
    # mp_state.stack: [batch_size, max_stack_size/num_time_steps, num_hiddens_for_stack]
    # mp_state.stack_pointers: [batch_size]
    stack, stack_pointers = mp_state.stack, mp_state.stack_pointers
    return stack.reshape(-1, stack.shape[-1])[stack_pointers + stack.shape[1] * jnp.arange(stack.shape[0]), :]

  def __call__(self, stack, stack_pointers, hint_preds, hiddens, graph_fts):
    """
    stack: [batch_size, max_stack_size/num_time_steps, num_hiddens_for_stack]
    stack_pointers: [batch_size]
    hiddens: [batch_size, num_nodes, hidden_dim]
    """
    # When num_hiddens_for_stack is set to same as hidden_dim (128), this reduces to Petar's suggestion.
    # Otherwise, it allows the algorithm to only save a subset of stuff to the stack and use the rest of the
    # embeddings just for predictions

    # [batch_size] containing [0 (pop), 1 (noop) or 2 (push)]
    # jax.debug.print("{x}", x=hint_preds["stack_op"])
    stack_ops = jnp.argmax(hint_preds["stack_op"], axis=-1)

    values = hiddens[:, :, :self.num_hiddens_for_stack]
    if self.value_network is not None:
      values = self.value_network(hiddens)
      # if self.print_counter % 1000 == 0:
      #   self.print_counter = 0
      #   jax.debug.print("new_stack_vals=\n{x}\n\nweights=\n{w}",
      #                   x=self.value_network(jnp.ones((1, 1, 128))),
      #                   w=self.value_network.layers[0].params_dict()["net/graph_level_callstack_module/~/linear/w"])
      # self.print_counter += 1

    if self.stack_pooling_fun is None:
      # turn [batch_size, dim] into [batch_size, num_nodes, dim]
      graph_fts = jnp.broadcast_to(graph_fts[:, None, :], (graph_fts.shape[0], hiddens.shape[1], graph_fts.shape[1]))
      att_in = jnp.concatenate((hiddens, graph_fts), axis=-1)
      # [batch_size, num_hiddens_for_stack] ((sum([batch_size, num_nodes, 1] * [batch_size, num_nodes, num_hiddens_for_stack]))
      new_stack_vals = jnp.sum(jax.nn.softmax(self.att_net(att_in), axis=-1) * values, axis=-2)
    else:
      # [batch_size, num_hiddens_for_stack] values that would be pushed to the stack in case of "push"
      new_stack_vals = self.stack_pooling_fun(values, axis=1)
    # values that will actually be on top of the stack in the next round
    # new_stack_vals = jnp.where(stack_ops[:, None] == StackOp.PUSH.value, new_stack_vals,
    #                            stack.reshape(-1, stack.shape[-1])[stack_pointers + 3 * jnp.arange(stack.shape[0]), :])
    # Overwrite the next stack element with the calculated one independent of the actual operation. The operation is
    # taken into account by only moving the pointers forward where we actually pushed
    stack = stack.reshape(-1, stack.shape[-1]).at[stack_pointers + 1 + stack.shape[1] * jnp.arange(stack.shape[0]), :] \
      .set(new_stack_vals) \
      .reshape(stack.shape)
    stack_pointers = jnp.maximum(stack_pointers + stack_ops - 1, 0)
    return stack, stack_pointers

  def add_to_features(self, top_stack, node_fts, edge_fts, graph_fts):
    if self.sum_fts is True:
      graph_fts = graph_fts + top_stack
    else:
      graph_fts = jnp.concatenate((graph_fts, top_stack), axis=-1)
    return node_fts, edge_fts, graph_fts

class NodeLevelCallstackModule(CallstackModule):
  def __init__(self, num_hiddens_for_stack: int, value_network: str, **kwargs):
    super().__init__(**kwargs)
    self.num_hiddens_for_stack = num_hiddens_for_stack
    if value_network is None or value_network == "":
      self.value_network = None
    else:
      self.value_network, output_dim = network_from_string(value_network)
      if output_dim != num_hiddens_for_stack:
        raise ValueError(
          f"Output dimension of the mlp ({output_dim}) must be equal to the stack dimension {num_hiddens_for_stack}!")

  def initial_values(self, batch_size: int, num_steps: int, num_nodes: int):
    stack = jnp.zeros((batch_size, num_steps + 1, num_nodes, self.num_hiddens_for_stack))
    stack_pointers = jnp.zeros((batch_size,), dtype=int)
    return stack, stack_pointers
  def get_top(self, mp_state: _MessagePassingScanState):
    # mp_state.stack: [batch_size, max_stack_size/num_time_steps, num_nodes, num_hiddens_for_stack]
    # mp_state.stack_pointers: [batch_size]
    stack, stack_pointers = mp_state.stack, mp_state.stack_pointers
    return stack.reshape(-1, stack.shape[2], stack.shape[3])[
           stack_pointers + jnp.arange(stack.shape[0]) * stack.shape[1], : , :]

  def __call__(self, stack, stack_pointers, hint_preds, hiddens, graph_fts):
    """
    stack: [batch_size, max_stack_size/num_time_steps, num_nodes, num_hiddens_for_stack]
    stack_pointers: [batch_size]
    """

    # [batch_size] containing [0 (pop), 1 (noop) or 2 (push)]
    stack_ops = jnp.argmax(hint_preds["stack_op"], axis=-1)
    # [batch_size, num_nodes, num_hiddens_for_stack] values that would be pushed to the stack in case of "push"
    if self.value_network is not None:
      hiddens = self.value_network(hiddens)
    new_stack_vals = hiddens[:, :, :, :self.num_hiddens_for_stack]

    # Overwrite the next stack element with the calculated one independent of the actual operation. The operation is
    # taken into account by only moving the pointers forward where we actually pushed
    stack = stack.reshape(-1, stack.shape[2], stack.shape[3])\
              .at[stack_pointers + 1 + stack.shape[1] * jnp.arange(stack.shape[0]), :, :]\
      .set(new_stack_vals) \
      .reshape(stack.shape)
    stack_pointers = jnp.maximum(stack_pointers + stack_ops - 1, 0)
    return stack, stack_pointers

  def add_to_features(self, top_stack, node_fts, edge_fts, graph_fts):
    graph_fts = jnp.concatenate((node_fts, top_stack), axis=-1)
    return node_fts, edge_fts, graph_fts

__all__ = [NoneCallstackModule, GraphLevelCallstackModule, NodeLevelCallstackModule]

CallstackFactory = Callable[[], CallstackModule]
def callstack_factory_from_name(callstack_type: str, **kwargs) -> CallstackFactory:
  for c in __all__:
    if c.__name__.lower()[:-(9+6)] == callstack_type.lower():
      return lambda: c(**kwargs)
  raise ValueError(f"There is no callstack type named {callstack_type}!")
