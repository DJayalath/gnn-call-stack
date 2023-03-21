from argparse import Namespace
from typing import List

import numpy as np
import wandb
def init(args):
  if args.use_wandb:
    kwargs = dict(project="gnn-call-stack", entity="camb-mphil", config=args)
    if args.wandb_name is not None:
      kwargs["name"] = f"{args.wandb_name}-{args.seed}"
    wandb.init(**kwargs)
    return wandb.config, wandb.config
  return args, args.flag_values_dict() # To make unpackable

def log(*args, **kwargs):
  if wandb.run is not None:
    wandb.log(*args, **kwargs)

def get_wandb_stats(run_names: List[str], metric_suffices: List[str]):
  api = wandb.Api()
  res = {suf: [] for suf in metric_suffices}
  names = []
  for run_name in run_names:
    run = api.run("camb-mphil/gnn-call-stack/" + run_name)
    history = run.scan_history()
    metric_names = {s: None for s in metric_suffices}
    missing_suffices = True
    vals = {suf: [] for suf in metric_suffices}
    for row in history:
      if missing_suffices:
        for k in row.keys():
          for suf in metric_suffices:
            if k.endswith(suf):
              metric_names[suf] = k
        missing_suffices = None in metric_names.values()
      for suffix, metric_name in metric_names.items():
        if metric_name in row and row[metric_name] is not None:
          vals[suffix].append(row[metric_name])
    names.append(run.config["wandb_name"])
    for k in vals.keys():
      res[k].append(np.array(vals[k]))
  if not all([n == names[0] for n in names]):
    print(names)
  return res, np.array(names) # just for our current use case

def wandb_mean_std(run_names, metric_suffices=["test.score.32", "test.score.96"]):
  res, names = get_wandb_stats(run_names, metric_suffices + ["val.score"])
  scores = {s: [] for s in metric_suffices}
  for run, val_scores in enumerate(res["val.score"]):
    best_val = np.argmax(val_scores)
    for s in metric_suffices:
      scores[s].append(res[s][run][best_val])
  print(f"{names[0].replace('_', ' ')} & " + " & ".join([f"{100*np.array(vals).mean():.2f}\\%$\\pm${100*np.array(vals).std():.2f}" for vals in scores.values()]) + "\\\\")