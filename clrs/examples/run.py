# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run training of one or more algorithmic tasks from CLRS."""

import re
import functools
import os
import shutil
import warnings
from typing import Any, Dict, List

from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf
import gnn_call_stack.utils as utils
from clrs._src import samplers
from clrs._src.samplers import SAMPLERS
from gnn_call_stack.callstacks import callstack_factory_from_name

# https://abseil.io/docs/python/guides/flags
flags.DEFINE_list('algorithms', ['dfs_callstack'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '8', '16', '24', '32'],
                  'Which training sizes to use. A size of -1 means using the predefined one.'
                  'use the benchmark dataset.')
flags.DEFINE_list('val_lengths', ['32'],
                  'Which test sizes to use.')
flags.DEFINE_list('test_lengths', ['-1'],
                  'Which test sizes to use.')
flags.DEFINE_string('sampler', 'default', # DfsTreeSampler, DfsSampler
                    'Allows to overwrite all data samplers with the given one.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 15000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 100, 'Evaluation frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'mpnn', # baseline: triplet_gmpnn
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', 'tmp/CLRS30',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')
flags.DEFINE_string('stack_pooling_fun', 'max', # ['max', 'min', 'mean', 'sum']
                  'Which pooling function to use for the node embeddings before pushing them on the stack. Can also '
                  'be a network specification as for value_network which ends in one dimension. In this case, the '
                  'given network will be applied to the hidden dimension, followed by a softmax to determine attention '
                  'weights for the different nodes.')
flags.DEFINE_integer('num_hiddens_for_stack', 64,
                    'How many of the node embedding entries to use for generating the stack embedding.')
flags.DEFINE_boolean('sum_fts', False,
                     'Whether to sum top stack embedding with graph features.')
flags.DEFINE_enum('callstack_type', 'graphlevel', ['none', 'graphlevel', 'nodelevel'],
                     'The type of callstack to use. This only works if the specification has a suitable hint called '
                     'stack_op.')
flags.DEFINE_string('value_network', '',
                    'Architecture of the MLP representing the value network of the callstack (e.g. 64_relu_128). '
                    'Numbers indicate linear layers and everything else the names of activation functions defined in '
                    'jax.nn.<function_name>. The final output dimension has to match num_hiddens for stack. If empty, '
                    'the first  num_hiddens_for_stack entries of the hidden state (e.g. for each node or pooled '
                    'according to stack_pooling_fun) will be taken directly.')
flags.DEFINE_boolean('use_recurrent_state', True,
                     'Whether to use the last node embeddings produced by the GNN of one algorithm\'s (time) step as ini')
flags.DEFINE_list('hints_to_output', [], # u_pi-pi[u]
                  'A list of assignments a-b[c] (read: a->b[c]) where <c> and <a> are expected to be graph-level node pointers. '
                  'They will be used to produce a node-level node-pointer output <b> by overwriting '
                  'the value of <b> at the position predicted by <c> with the value predicted in <a> in every time '
                  'step.')
flags.DEFINE_boolean('checkpoint_wandb', True,
                     'Whether to save the checkpoint files to weights and biases.')
flags.DEFINE_boolean('use_wandb', True,
                     'Whether to log to weights and biases.')
flags.DEFINE_string('wandb_name', None,
                    'The name of the wandb run. Automatically generated by default.')

FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
  """Download CLRS30 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    # logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    # return dataset_folder
    logging.info('Dataset found at %s. Deleting and re-generating to be sure distribution is correct.', dataset_folder)
    shutil.rmtree(dataset_path)
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
  """Create a sampler with given options.

  Args:
    length: Size of samples (i.e., number of nodes in the graph).
      A length of -1 will mean that the benchmark
      dataset (for the given split) is used. Positive sizes will instantiate
      samplers of the corresponding size.
    rng: Numpy random state.
    algorithm: The name of the algorithm to sample from.
    split: 'train', 'val' or 'test'.
    batch_size: Samples per batch.
    multiplier: Integer multiplier for the number of samples in the dataset,
      only used for positive sizes. Negative multiplier means infinite samples.
    randomize_pos: Whether to randomize the `pos` input.
    enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
    enforce_permutations: Whether to enforce permutation pointers.
    chunked: Whether to chunk the dataset.
    chunk_length: Unroll length of chunks, if `chunked` is True.
    sampler_kwargs: Extra args passed to the sampler.
  Returns:
    A sampler (iterator), the number of samples in the iterator (negative
    if infinite samples), and the spec.
  """
  if length < 0:  # load from file
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32, dtype=np.uint32),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
        )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  """Create a sampler with cycling sample sizes."""
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)

# predict_fn: basically BaselineModel.predict() (baselines.py)
def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  # feedback.features.hints[7 | "time"] suggests that the dimensions are [time, batch, ...] where the same time can
  # occur multiple times, the last one might be < 1.0 and
  # The sample includes all the time steps so this is where we iterate over them
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def create_samplers(rng, train_lengths: List[int], val_lengths: List[int], test_lengths: List[int]):
  """Create all the samplers."""
  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []

  for algo_idx, algorithm in enumerate(FLAGS.algorithms):
    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):

      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        # Fixed haystack + needle; variability will be in needle
        # Still, for chunked training, we maintain as many samplers
        # as train lengths, since, for each length there is a separate state,
        # and we must keep the 1:1 relationship between states and samplers.
        max_length = max(train_lengths)
        if max_length > 0:  # if < 0, we are using the benchmark data
          max_length = (max_length * 5) // 4
        train_lengths = [max_length]
        if FLAGS.chunked_training:
          train_lengths = train_lengths * len(train_lengths)

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        # Choose a lower connection probability for the above algorithms,
        # otherwise trajectories are very long
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=FLAGS.algorithms[algo_idx],
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
          chunk_length=FLAGS.chunk_length,
          )

      train_args = dict(sizes=train_lengths,
                        split='train',
                        batch_size=FLAGS.batch_size,
                        multiplier=-1,
                        randomize_pos=FLAGS.random_pos,
                        chunked=FLAGS.chunked_training,
                        sampler_kwargs=sampler_kwargs,
                        **common_sampler_args)
      train_sampler, _, spec = make_multi_sampler(**train_args)

      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      val_args = dict(sizes=val_lengths,
                      split='val',
                      batch_size=32,
                      multiplier=2 * mult,
                      randomize_pos=FLAGS.random_pos,
                      chunked=False,
                      sampler_kwargs=sampler_kwargs,
                      **common_sampler_args)
      val_sampler, val_samples, spec = make_multi_sampler(**val_args)

      for length in test_lengths:
        test_args = dict(sizes=[length],
                         split='test',
                         batch_size=32,
                         multiplier=2 * mult,
                         randomize_pos=False,
                         chunked=False,
                         sampler_kwargs={},
                         **common_sampler_args)
        test_sampler, test_samples, spec = make_multi_sampler(**test_args)
        test_samplers.append(test_sampler)
        test_sample_counts.append(test_samples)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)


  return (train_samplers,
          val_samplers, val_sample_counts,
          [test_samplers], [test_sample_counts],
          spec_list)


def main(unused_argv):
  global FLAGS
  # FLAGS.use_callstack = FLAGS.callstack_type != 'none' # for backwards compatibility of logs
  FLAGS, unpackable_flags = utils.init(FLAGS)
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]
  val_lengths = [int(x) for x in FLAGS.val_lengths]
  test_lengths = [int(x) for x in FLAGS.test_lengths]

  if FLAGS.sampler != "" and FLAGS.sampler.lower() != "default":
    if -1 in train_lengths or -1 in test_lengths or -1 in val_lengths:
      warnings.warn(f"You are using the custom data sampler {FLAGS.sampler} but some of your samplers include length "
                    f"-1 which means they come from a pre-defined dataset with possibly different distribution!")
    sampler = getattr(samplers, FLAGS.sampler)
    for k in samplers.SAMPLERS:
      samplers.SAMPLERS[k] = sampler

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32, dtype=np.uint32))

  # Create samplers
  (train_samplers,
   val_samplers, val_sample_counts,
   test_samplers_all, test_sample_counts_all,
   spec_list) = create_samplers(rng, train_lengths, val_lengths, test_lengths)

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads
  )
  def process_hint_to_output(spec: str):
    res = re.match("(.+)-(.+)\\[(.+)\\]", spec)
    if not res:
      raise ValueError(f"Hint to output specification {spec} does not match pattern a-b[c]!")
    return res.group(1), res.group(2), res.group(3)
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      checkpoint_wandb=FLAGS.checkpoint_wandb,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      callstack_factory=callstack_factory_from_name(**unpackable_flags),
      use_recurrent_state=FLAGS.use_recurrent_state,
      hints_to_output=[process_hint_to_output(h2o) for h2o in FLAGS.hints_to_output]
      )

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in train_samplers],
        **model_params
        )
  else:
    train_model = eval_model

  # Training loop.
  best_score = -1.0
  current_train_items = [0] * len(FLAGS.algorithms)
  step = 0
  next_eval = 0
  next_test = 0
  # Make sure scores improve on first step, but not overcome best score
  # until all algos have had at least one evaluation.
  val_scores = [-99999.9] * len(FLAGS.algorithms)
  length_idx = 0

  while step < FLAGS.train_steps:
    feedback_list = [next(t) for t in train_samplers]

    # Initialize model.
    if step == 0:
      all_features = [f.features for f in feedback_list]
      if FLAGS.chunked_training:
        # We need to initialize the model with samples of all lengths for
        # all algorithms. Also, we need to make sure that the order of these
        # sample sizes is the same as the order of the actual training sizes.
        all_length_features = [all_features] + [
            [next(t).features for t in train_samplers]
            for _ in range(len(train_lengths))]
        train_model.init(all_length_features[:-1], FLAGS.seed + 1)
      else:
        train_model.init(all_features, FLAGS.seed + 1)

    # Training step.
    for algo_idx in range(len(train_samplers)):
      feedback = feedback_list[algo_idx]
      rng_key, new_rng_key = jax.random.split(rng_key)
      if FLAGS.chunked_training:
        # In chunked training, we must indicate which training length we are
        # using, so the model uses the correct state.
        length_and_algo_idx = (length_idx, algo_idx)
      else:
        # In non-chunked training, all training lengths can be treated equally,
        # since there is no state to maintain between batches.
        length_and_algo_idx = algo_idx
      cur_loss = train_model.feedback(rng_key, feedback, length_and_algo_idx)
      rng_key = new_rng_key

      if FLAGS.chunked_training:
        examples_in_chunk = np.sum(feedback.features.is_last).item()
      else:
        examples_in_chunk = len(feedback.features.lengths)
      current_train_items[algo_idx] += examples_in_chunk
      # to compare results with the standard 32-batch_size experiments
      # logging.info('Algo %s step %i current loss %f, current_train_items %i.',
      #              FLAGS.algorithms[algo_idx], step,
      #              cur_loss, current_train_items[algo_idx])
      utils.log({FLAGS.algorithms[algo_idx]: {
        "train_loss": cur_loss
        #"training_items": current_train_items
      }}, step=step)

    # Periodically evaluate model
    if step >= next_eval:
      eval_model.params = train_model.params
      for algo_idx in range(len(train_samplers)):
        common_extras = {'examples_seen': current_train_items[algo_idx],
                         'step': step,
                         'algorithm': FLAGS.algorithms[algo_idx]}

        # Validation info.
        new_rng_key, rng_key = jax.random.split(rng_key)
        val_stats = collect_and_eval(
            val_samplers[algo_idx],
            functools.partial(eval_model.predict, algorithm_index=algo_idx),
            val_sample_counts[algo_idx],
            new_rng_key,
            extras=common_extras)
        # logging.info('(val) algo %s step %d: %s',
        #              FLAGS.algorithms[algo_idx], step, val_stats)
        utils.log({FLAGS.algorithms[algo_idx]:
                     {"val." + k: v for k, v in val_stats.items() if k not in ["step", "algorithm"]}}, step=step)
        val_scores[algo_idx] = val_stats['score']

      next_eval += FLAGS.eval_every

      # If best total score, update best checkpoint.
      # Also save a best checkpoint on the first step.
      msg = (f'best avg val score was '
             f'{best_score/len(FLAGS.algorithms):.3f}, '
             f'current avg val score is {np.mean(val_scores):.3f}, '
             f'val scores are: ')
      msg += ', '.join(
          ['%s: %.3f' % (x, y) for (x, y) in zip(FLAGS.algorithms, val_scores)])
      if (sum(val_scores) > best_score) or step == 0:
        best_score = sum(val_scores)
        logging.info('Checkpointing best model, %s', msg)
        train_model.save_model('best.pkl')
      else:
        logging.info('Not saving new best model, %s', msg)

    if step >= next_test:
      for algo_idx in range(len(train_samplers)):
        common_extras = {'examples_seen': current_train_items[algo_idx],
                         'step': step,
                         'algorithm': FLAGS.algorithms[algo_idx]}
        log_dict = {"test": {"score": {}}}
        for i in range(len(test_samplers_all[algo_idx])):
          new_rng_key, rng_key = jax.random.split(rng_key)
          test_stats = collect_and_eval(
            test_samplers_all[algo_idx][i],
            functools.partial(eval_model.predict, algorithm_index=algo_idx),
            test_sample_counts_all[algo_idx][i],
            new_rng_key,
            extras=common_extras)
          logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)
          log_dict["test"]["score"][test_lengths[i]] = test_stats["score"]

        utils.log({FLAGS.algorithms[algo_idx]: log_dict}, step=step)
        next_test += FLAGS.test_every

    step += 1
    length_idx = (length_idx + 1) % len(train_lengths)

  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)

  for algo_idx in range(len(train_samplers)):
    common_extras = {'examples_seen': current_train_items[algo_idx],
                     'step': step,
                     'algorithm': FLAGS.algorithms[algo_idx]}
    log_dict = {"test": {"score": {}}}
    for i in range(len(test_samplers_all[algo_idx])):
      new_rng_key, rng_key = jax.random.split(rng_key)
      test_stats = collect_and_eval(
        test_samplers_all[algo_idx][i],
        functools.partial(eval_model.predict, algorithm_index=algo_idx),
        test_sample_counts_all[algo_idx][i],
        new_rng_key,
        extras=common_extras)
      logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)
      log_dict["test"]["score"][test_lengths[i]] = test_stats["score"]

    utils.log({FLAGS.algorithms[algo_idx]: log_dict}, step=step)

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
