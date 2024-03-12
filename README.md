# Recursive Algorithmic Reasoning

This work is based on the [CLRS30 algorithmic reasoning benchmark](https://github.com/deepmind/clrs) and builds GNNs with call stacks for solving recursive algorithmic problems. It was published as a full conference paper + oral (top 5%) at Learning on Graphs (LoG) 2023 and first featured at the Knowledge and Logical Reasoning (KLR) workshop at ICML 2023. You can read the full paper [here](https://openreview.net/forum?id=43M1bPorxU). If you found our work helpful in your research, please cite

```
@inproceedings{juerss2023recursive,
    title = {Recursive Algorithmic Reasoning},
    author = {Jonas J{\"{u}}r{\ss} and
              Dulhan Jayalath and
              Petar Veli\v{c}kovi\'{c}},
    booktitle = {The Second Learning on Graphs Conference},
    year = {2023},
    url = {https://openreview.net/forum?id=43M1bPorxU}
}
```

## Reproducing Experiments from "Recursive Algorithmic Reasoning" (LoG 2023)
In our experiments, we always use the random seeds 1-3.
### Ours
```
python -m clrs.examples.run --train_steps 20000 --callstack_type graphlevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --nouse_recurrent_state --wandb_name ours --seed <seed>
```
### No Stack
```
python -m clrs.examples.run --train_steps 20000 --callstack_type none --algorithms dfs_callstack_localhints_no_stackhint --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --nouse_recurrent_state --test_lengths 32,96 --wandb_name no_stack-no_stack_hints --seed <seed>
```

### With hidden state
```
python -m clrs.examples.run --train_steps 20000 --callstack_type graphlevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --wandb_name recurrent --seed <seed>
```

### No stack and hidden state
```
python -m clrs.examples.run --train_steps 20000 --callstack_type none --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --wandb_name no_stack-recurrent --seed <seed>
```

### No output collection
```
python -m clrs.examples.run --train_steps 20000 --callstack_type graphlevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --test_lengths 32,96 --nouse_recurrent_state --wandb_name no_collection --seed <seed>
```

### No teacher forcing
```
python -m clrs.examples.run --train_steps 20000 --callstack_type graphlevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --nouse_recurrent_state --wandb_name no_tf --seed <seed>
```

### No value network
```
python -m clrs.examples.run --train_steps 20000 --callstack_type graphlevel --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --nouse_recurrent_state --wandb_name no_valuenet --seed <seed>
```

### With attention
```
python -m clrs.examples.run --train_steps 20000 --callstack_type graphlevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --stack_pooling_fun 128_relu_1 --nouse_recurrent_state --wandb_name attention --seed <seed>
```

### As in original Generalist Reasoner paper
```
python -m clrs.examples.run --train_steps 20000 --callstack_type none --value_network 128_relu_64 --algorithms dfs --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --test_lengths 32,96 --wandb_name orignal_dfs --seed <seed>
```

### Node-level stack
```
python -m clrs.examples.run --train_steps 20000 --callstack_type nodelevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --nouse_recurrent_state --wandb_name nodelevel --seed <seed>
```

### Node-level stack + recurrent state
```
python -m clrs.examples.run --train_steps 20000 --callstack_type nodelevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32,96 --wandb_name nodelevel-recurrent --seed <seed>
```

## Reproducing Experiments from "Recursive Reasoning with Neural Networks" (ICLR 2023, Tiny Paper)

To reproduce our results, please first follow the setup instructions in the [CLRS30 algorithmic reasoning benchmark](https://github.com/deepmind/clrs) and checkout commit `3ed18e8`.

### With callstack (ours)
```
python -m clrs.examples.run --callstack_type graphlevel --value_network 128_relu_64 --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --test_lengths 32 48 --nouse_recurrent_state --seed <seed>
```

### Only stack operation hint
```
python -m clrs.examples.run --callstack_type none --algorithms dfs_callstack_localhints --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --nouse_recurrent_state --test_lengths 32 48 --seed <seed>
```

### Neither callstack nor stack operation hint
```
python -m clrs.examples.run --callstack_type none --algorithms dfs_callstack_localhints_no_stackhint --hint_teacher_forcing 0.5 --sampler DfsMixedTreeSampler --hints_to_output u_pi-pi[u] --nouse_recurrent_state --test_lengths 32 48 --seed <seed>
```
