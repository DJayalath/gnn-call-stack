from argparse import Namespace

import wandb
def init(args):
  if args.use_wandb:
    kwargs = dict(project="gnn-call-stack", entity="camb-mphil", config=args)
    if args.wandb_name is not None:
      kwargs["name"] = args.wandb_name
    wandb.init(**kwargs)
    return wandb.config, wandb.config
  return args, args.flag_values_dict() # To make unpackable

def log(*args, **kwargs):
  if wandb.run is not None:
    wandb.log(*args, **kwargs)
