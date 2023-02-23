from __future__ import annotations

import abc
from typing import Callable

import jax.numpy as jnp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from clrs._src.nets import _MessagePassingScanState


class Callstack(abc.ABC):

  def __init__(self, **kwargs):
    pass

  @abc.abstractmethod
  def initial_values(self, batch_size: int, num_steps: int, num_nodes: int):
    pass
  @abc.abstractmethod
  def get_top(self, mp_state: _MessagePassingScanState):
    pass

  @abc.abstractmethod
  def step(self, mp_state, hint_preds, hiddens):
    pass

  @abc.abstractmethod
  def add_to_features(self, top_stack, node_fts, edge_fts, graph_fts):
    pass

class NoneCallstack(Callstack):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def initial_values(self, batch_size: int, num_steps: int, num_nodes: int):
    return None, None
  def get_top(self, mp_state: _MessagePassingScanState):
    return None
  def step(self, mp_state, hint_preds, hiddens):
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


class GraphLevelCallstack(Callstack):

  def __init__(self, num_hiddens_for_stack: int, stack_pooling_fun: str, **kwargs):
    super().__init__(**kwargs)
    self.num_hiddens_for_stack = num_hiddens_for_stack
    self.stack_pooling_fun = getattr(jnp, stack_pooling_fun)

  def initial_values(self, batch_size: int, num_steps: int, num_nodes: int):
    stack = jnp.zeros((batch_size, num_steps + 1, self.num_hiddens_for_stack))
    stack_pointers = jnp.zeros((batch_size,), dtype=int)
    return stack, stack_pointers
  def get_top(self, mp_state: _MessagePassingScanState):
    # mp_state.stack: [batch_size, max_stack_size/num_time_steps, num_hiddens_for_stack]
    # mp_state.stack_pointers: [batch_size]
    stack, stack_pointers = mp_state.stack, mp_state.stack_pointers
    return stack.reshape(-1, stack.shape[-1])[stack_pointers + stack.shape[1] * jnp.arange(stack.shape[0]), :]

  def step(self, mp_state, hint_preds, hiddens):
    # [batch_size, max_stack_size/num_time_steps, num_hiddens_for_stack]
    stack = mp_state.stack
    # [batch_size]
    stack_pointers = mp_state.stack_pointers

    # When num_hiddens_for_stack is set to same as hidden_dim (128), this reduces to Petar's suggestion.
    # Otherwise, it allows the algorithm to only save a subset of stuff to the stack and use the rest of the
    # embeddings just for predictions

    # [batch_size] containing [0 (pop), 1 (noop) or 2 (push)]
    stack_ops = jnp.argmax(hint_preds["stack_op"], axis=-1)
    # [batch_size, num_hiddens_for_stack] values that would be pushed to the stack in case of "push"
    new_stack_vals = self.stack_pooling_fun(hiddens[:, :, :self.num_hiddens_for_stack], axis=1)
    # values that will actually be on top of the stack in the next round
    # new_stack_vals = jnp.where(stack_ops[:, None] == StackOp.PUSH.value, new_stack_vals,
    #                            stack.reshape(-1, stack.shape[-1])[stack_pointers + 3 * jnp.arange(stack.shape[0]), :])
    # Overwrite the next stack element with the calculated one independent of the actual operation. The operation is
    # taken into account by only moving the pointers forward where we actually pushed
    stack = stack.reshape(-1, stack.shape[-1]).at[stack_pointers + 1 + stack.shape[1] * jnp.arange(stack.shape[0]), :] \
      .set(new_stack_vals) \
      .reshape(mp_state.stack.shape)
    stack_pointers = jnp.maximum(stack_pointers + stack_ops - 1, 0)
    return stack, stack_pointers

  def add_to_features(self, top_stack, node_fts, edge_fts, graph_fts):
    graph_fts = jnp.concatenate((graph_fts, top_stack), axis=-1)
    return node_fts, edge_fts, graph_fts

class NodeLevelCallstack(Callstack):

  def __init__(self, num_hiddens_for_stack: int, **kwargs):
    super().__init__(**kwargs)
    self.num_hiddens_for_stack = num_hiddens_for_stack

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

  def step(self, mp_state, hint_preds, hiddens):
    # [batch_size, max_stack_size/num_time_steps, num_nodes, num_hiddens_for_stack]
    stack = mp_state.stack
    # [batch_size]
    stack_pointers = mp_state.stack_pointers

    # [batch_size] containing [0 (pop), 1 (noop) or 2 (push)]
    stack_ops = jnp.argmax(hint_preds["stack_op"], axis=-1)
    # [batch_size, num_nodes, num_hiddens_for_stack] values that would be pushed to the stack in case of "push"
    new_stack_vals = hiddens[:, :, :, :self.num_hiddens_for_stack]

    # Overwrite the next stack element with the calculated one independent of the actual operation. The operation is
    # taken into account by only moving the pointers forward where we actually pushed
    stack = stack.reshape(-1, stack.shape[2], stack.shape[3])\
              .at[stack_pointers + 1 + stack.shape[1] * jnp.arange(stack.shape[0]), :, :]\
      .set(new_stack_vals) \
      .reshape(mp_state.stack.shape)
    stack_pointers = jnp.maximum(stack_pointers + stack_ops - 1, 0)
    return stack, stack_pointers

  def add_to_features(self, top_stack, node_fts, edge_fts, graph_fts):
    graph_fts = jnp.concatenate((node_fts, top_stack), axis=-1)
    return node_fts, edge_fts, graph_fts

__all__ = [NoneCallstack, GraphLevelCallstack, NodeLevelCallstack]
def callstack_from_name(callstack_type: str, **kwargs) -> Callstack:
  for c in __all__:
    if c.__name__.lower()[:-9] == callstack_type.lower():
      return c(**kwargs)
  raise ValueError(f"There is no callstack type named {callstack_type}!")
