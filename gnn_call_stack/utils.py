from __future__ import annotations

import csv
import warnings
from typing import List
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from clrs import DataPoint

import wandb
def init(args):
  if args.use_wandb:
    wandb.init(project="gnn-call-stack", entity="camb-mphil", config=args)
    return wandb.config, wandb.config
  return args, args.flag_values_dict() # To make unpackable

def log(*args, **kwargs):
  if wandb.run is not None:
    wandb.log(*args, **kwargs)

GRAPHS_TO_PLOT: int = 0
ITERATION: int = 0

__lock__ = False

def log_graph(time_step: np.ndarray, adj, hints: List[DataPoint], hint_preds: dict):
  # super naive semaphore but does it's job for now
  global __lock__
  while __lock__:
    pass
  if GRAPHS_TO_PLOT == 0:
    return
  __lock__ = True
  """
  adj: [nb_nodes, nb_nodes]
  returns: (node_table: wandb.Table, edge_table: wandb.Table)
  """
  time_step = time_step.item()
  from clrs import Location, Type
  if len(adj.shape) != 3 or adj.shape[1] != adj.shape[2]:
    raise ValueError(f"Invalid adjacency matrix shape: {adj.shape}!")
  nb_nodes = adj.shape[1]

  graph_hints = [dp for dp in hints if dp.location == Location.GRAPH]
  edge_hints = [dp for dp in hints if dp.location == Location.EDGE]
  node_hints = [dp for dp in hints if dp.location == Location.NODE and dp.type_ not in [Type.POINTER, Type.SOFT_POINTER]]
  node_to_node_hints = [(dp, dp.data) for dp in hints if dp.location == Location.NODE and dp.type_ == Type.POINTER]
  node_to_node_hints += [(dp, np.argmax(dp.data, axis=-1)) for dp in hints if dp.location == Location.NODE and dp.type_ == Type.SOFT_POINTER]
  if edge_hints:
    warnings.warn('Ignored edge-level hints when plotting graph as this is not implemented yet.', UserWarning)

  with open(f'graph-logs/nodes-{ITERATION}-{time_step}.csv', 'w', newline='') as nodes_file,\
      open(f'graph-logs/edges-{ITERATION}-{time_step}.csv', 'w', newline='') as edges_file,\
      open(f'graph-logs/pointers-{ITERATION}-{time_step}.csv', 'w', newline='') as pointers_file,\
      open(f'graph-logs/graph-{ITERATION}-{time_step}.txt', 'w', newline='') as graph_file:
    nodes_writer = csv.writer(nodes_file)
    nodes_writer.writerow(["graph", "pool_step", "node_index", "r", "g", "b", "label", "activations"])
    edges_writer = csv.writer(edges_file)
    edges_writer.writerow(["graph", "pool_step", "source", "target", "strength", "label"])
    pointers_writer = csv.writer(pointers_file)
    pointers_writer.writerow(["graph", "pool_step", "source", "target", "name", "prediction"])

    for num_graph in range(GRAPHS_TO_PLOT):
      for i in range(nb_nodes):
        node_label = ", ".join([f"{val.name}: {np.array2string(val.data[num_graph, i], precision=2)} "
                                f"({np.array2string(hint_preds[val.name][num_graph, i], precision=2)})"
                                for val in node_hints])
        color = next((dp.data[num_graph, i] for dp in node_hints if dp.name == "color"), None)
        if color is None:
          color = np.array([0, 0, 255])
        else:
          color = np.array([1, 1, 1]) * (255 * color[0] + 127 * color[1])
        nodes_writer.writerow([num_graph, time_step, i, *color, node_label, ""])
        for pointer, target_nodes in node_to_node_hints:
          # pred_label = ", ".join([f"{val:.2f}" for val in hint_preds[pointer.name][num_graph, i, :]])
          pred_label = f"{pointer.name}: {np.array2string(pointer.data[num_graph, i], precision=2)} ({np.array2string(hint_preds[pointer.name][num_graph, i], precision=2)})"
          pointers_writer.writerow([num_graph, time_step, i, target_nodes[num_graph, i], pointer.name, pred_label])
        for j in range(nb_nodes):
          # TODO: check if that's the right way around or if it should be adj[j, i]
          if adj[num_graph, i, j] != 0:
            edges_writer.writerow([num_graph, time_step, i, j, adj[num_graph, i, j], ""])

    graph_file.write("\n".join([f"{val.name}: {np.array2string(val.data, precision=2)} "
                                f"({np.array2string(hint_preds[val.name], precision=2)})"
                                for val in graph_hints]))
    nodes_file.flush()
    edges_file.flush()
    pointers_file.flush()
    graph_file.flush()
  __lock__ = False
def log_graph_wandb(time_step: np.ndarray, adj, hints: List[DataPoint], hint_preds: dict):
  """
  adj: [nb_nodes, nb_nodes]
  returns: (node_table: wandb.Table, edge_table: wandb.Table)
  """
  time_step = time_step.item()
  from clrs import Location, Type
  if len(adj.shape) != 3 or adj.shape[1] != adj.shape[2]:
    raise ValueError(f"Invalid adjacency matrix shape: {adj.shape}!")
  nb_nodes = adj.shape[1]

  graph_hints = [dp for dp in hints if dp.location == Location.GRAPH]
  edge_hints = [dp for dp in hints if dp.location == Location.EDGE]
  node_hints = [dp for dp in hints if dp.location == Location.NODE and dp.type_ != Type.POINTER]
  node_to_node_hints = [dp for dp in hints if dp.location == Location.NODE and dp.type_ == Type.POINTER]
  if edge_hints:
    warnings.warn('Ignored edge-level hints when plotting graph as this is not implemented yet.', UserWarning)

  node_table = wandb.Table(["graph", "pool_step", "node_index", "r", "g", "b", "label", "activations"])
  edge_table = wandb.Table(["graph", "pool_step", "source", "target", "strength", "label"])
  pointer_table = wandb.Table(["graph", "pool_step", "source", "target", "name", "prediction"])
  for num_graph in range(GRAPHS_TO_PLOT):
    for i in range(nb_nodes):
      node_label = ", ".join([f"{val.name}: {val.data[num_graph, i]} ({hint_preds[val.name][num_graph, i]})" for val in node_hints])
      node_table.add_data(num_graph, time_step, i, 0, 0, 0, node_label, "")
      for pointer in node_to_node_hints:
        pred_label = ", ".join([f"{val:.2f}" for val in hint_preds[pointer.name][num_graph, i, :]])
        pointer_table.add_data(num_graph, time_step, i, pointer.data[num_graph, i], pointer.name, pred_label)
      for j in range(nb_nodes):
        # TODO: check if that's the right way around or if it should be adj[j, i]
        if adj[num_graph, i, j] != 0:
          edge_table.add_data(num_graph, time_step, i, j, adj[num_graph, i, j], "")

  log(dict(
    node_table=node_table,
    edge_table=edge_table,
    pointer_table=pointer_table
  ), step=ITERATION)

# def adj_to_wandb_tables(adj: np.ndarray, time_step: int):
#   """
#   adj: [nb_nodes, nb_nodes]
#   returns: (node_table: wandb.Table, edge_table: wandb.Table)
#   """
#   global num_graph
#   if len(adj.shape) != 2 or adj.shape[0] != adj.shape[1]:
#     raise ValueError(f"Invalid adjacency matrix shape: {adj.shape}!")
#   nb_nodes = adj.shape[0]
#   node_table = wandb.Table(["graph", "pool_step", "node_index", "r", "g", "b", "label", "activations"])
#   edge_table = wandb.Table(["graph", "pool_step", "source", "target", "strength", "label"])
#   for i in range(nb_nodes):
#     node_table.add_row([num_graph, time_step, i, 0, 0, 0, "", ""])
#     for j in range(nb_nodes):
#       # TODO: check if that's the right way around or if it should be adj[j, i]
#       if adj[i, j] != 0:
#         edge_table.add_row([num_graph, time_step, i, j, adj[i, j], ""])
#
#   num_graph += 1
#   return node_table, edge_table
