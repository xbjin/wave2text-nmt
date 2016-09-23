"""Binary for training translation models and decoding from them

See the following papers for more information on neural translation models
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import argparse
import subprocess
import tensorflow as tf
import yaml
import random
import numpy as np

from pprint import pformat
from operator import itemgetter
from translate import utils
from translate.multitask_model import MultiTaskModel


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
parser.add_argument('--reset', help='reset model (don\'t load any checkpoint)', action='store_true')
parser.add_argument('--reset-learning-rate', help='reset learning rate', action='store_true')

# Available actions (exclusive)
parser.add_argument('--decode', help='translate this corpus')
parser.add_argument('--eval', help='compute BLEU score on this corpus')
parser.add_argument('--train', help='train an NMT model', action='store_true')
parser.add_argument('config', help='load a configuration file in the YAML format')

# Tensorflow configuration
parser.add_argument('--gpu-id', type=int, help='index of the GPU where to run the computation')
parser.add_argument('--no-gpu', action='store_true', help='run on CPU')

# Decoding options (to avoid having to edit the config file)
parser.add_argument('--beam-size', type=int)
parser.add_argument('--ensemble', action='store_const', const=True)
parser.add_argument('--lm-file')
parser.add_argument('--checkpoints', nargs='+')
parser.add_argument('--lm-weight', type=float)
parser.add_argument('--output')
parser.add_argument('--max-steps', type=int)
parser.add_argument('--remove-unk', action='store_const', const=True)


"""
Random thoughts
---------------

data: http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/

Features:
- copy vocab and config to model dir
- rename scopes to nicer names + do mapping of existing models
- time pooling: concat or sum instead of skipping

Benchmarks:
- compare our baseline system with vanilla Tensorflow seq2seq, and GroundHog/blocks-examples
- replicate Jean et al. (2015)'s results
- analyze the impact of this initial_state_attention parameter
- replicate the experiments of the WMT paper on neural post-editing
- test convolutional attention (on speech recognition)

Evaluation with METEOR:
java -jar scripts/meteor-1.5.jar {hyp} {ref} -l {trg_ext} -a ~servan/Tools/METEOR/data/paraphrase-en.gz
"""


def main(args=None):
  args = parser.parse_args(args)

  # read config file and default config
  with open('config/default.yaml') as f:
    default_config = utils.AttrDict(yaml.safe_load(f))

  with open(args.config) as f:
    config = utils.AttrDict(yaml.safe_load(f))
    # command-line parameters have higher precedence than config file
    for k, v in vars(args).items():
      if v is not None and (k in default_config or k in ('decode', 'eval', 'output')):
        config[k] = v

    # set default values for parameters that are not defined
    for k, v in default_config.items():
      config.setdefault(k, v)

  # enforce parameter constraints
  assert config.steps_per_eval % config.steps_per_checkpoint == 0, (
    'steps-per-eval should be a multiple of steps-per-checkpoint')
  assert args.decode or args.eval or args.train, (
    'you need to specify at least one action (decode, eval, or train)')
  assert args.train or 'tasks' not in args or len(args.tasks) == 1, (
    'you cannot set multiple tasks in decode and eval modes')

  logging_level = logging.DEBUG if args.verbose else logging.INFO
  # always log to stdout in decoding and eval modes (to avoid overwriting precious train logs)
  logger = utils.create_logger(config.log_file if args.train else None)
  logger.setLevel(logging_level)

  utils.log(' '.join(sys.argv))  # print command line
  try:                           # print git hash
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    utils.log('commit hash {}'.format(commit_hash))
  except:
    pass

  # list of encoder and decoder parameter names (each encoder and decoder can have a different value
  # for those parameters)
  model_parameters = [
    'cell_size', 'layers', 'vocab_size', 'embedding_size', 'attention_filters', 'attention_filter_length',
    'use_lstm', 'time_pooling', 'attention_window_size', 'dynamic', 'binary', 'character_level', 'bidir',
    'load_embeddings', 'pooling_avg'
  ]
  # TODO: independent model dir for each task
  task_parameters = ['data_dir', 'train_prefix', 'dev_prefix', 'vocab_prefix', 'ratio',
                     'lm_file', 'learning_rate', 'learning_rate_decay_factor', 'max_output_len']

  # in case no task is defined (standard mono-task settings), define a "main" task
  config.setdefault('tasks', [{'encoders': config.encoders, 'decoder': config.decoder, 'name': 'main', 'ratio': 1.0}])
  config.tasks = [utils.AttrDict(task) for task in config.tasks]

  for task in config.tasks:
    # convert dicts to AttrDicts for convenience
    task.encoders = [utils.AttrDict(encoder) for encoder in task.encoders]
    task.decoder = utils.AttrDict(task.decoder)

    for parameter in task_parameters:
      task.setdefault(parameter, config.get(parameter))
    for encoder_or_decoder in task.encoders + [task.decoder]:
      # move parameters all the way up from base level to encoder/decoder level:
      # default values for encoder/decoder parameters can be defined at the task level and base level
      # default values for tasks can be defined at the base level
      for parameter in model_parameters:
        if parameter in encoder_or_decoder:
          continue
        elif parameter in task:
          encoder_or_decoder[parameter] = task[parameter]
        else:
          encoder_or_decoder[parameter] = config.get(parameter)

  # log parameters
  utils.log('program arguments')
  for k, v in sorted(config.items(), key=itemgetter(0)):
    if k == 'tasks':
      utils.log('  {:<20}\n{}'.format(k, pformat(v)))
    elif k not in model_parameters and k not in task_parameters:
      utils.log('  {:<20} {}'.format(k, pformat(v)))
  
  device = None
  if args.no_gpu:
    device = '/cpu:0'
  elif args.gpu_id is not None:
    device = '/gpu:{}'.format(args.gpu_id)
  
  utils.log('creating model')
  utils.log('using device: {}'.format(device))

  with tf.device(device):
    checkpoint_dir = os.path.join(config.model_dir, 'checkpoints')
    initializer = None  # default initializer
    # all parameters except source embeddings and bias variables are initialized with this
    # initializer = tf.random_normal_initializer(stddev=0.1)   # TODO: try this one
    with tf.variable_scope('seq2seq', initializer=initializer):
      model = MultiTaskModel(name='main', checkpoint_dir=checkpoint_dir, **config)

  utils.log('model parameters ({})'.format(len(tf.all_variables())))
  for var in tf.all_variables():
    utils.log('  {} shape {}'.format(var.name, var.get_shape()))

  tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  tf_config.gpu_options.allow_growth = config.allow_growth
  tf_config.gpu_options.per_process_gpu_memory_fraction = config.mem_fraction

  with tf.Session(config=tf_config) as sess:
    best_checkpoint = os.path.join(checkpoint_dir, 'best')

    if config.ensemble and (args.eval or args.decode):
      # create one session for each model in the ensemble
      sess = [tf.Session() for _ in config.checkpoints]
      for sess_, checkpoint in zip(sess, config.checkpoints):
        model.initialize(sess_, [checkpoint], reset=True)
    elif not config.checkpoints and not args.reset and (args.eval or args.decode) and os.path.isfile(best_checkpoint):
      # in decoding and evaluation mode, unless specified otherwise (by `checkpoints` or `reset` parameters,
      # try to load the best checkpoint)
      model.initialize(sess, [best_checkpoint], reset=True)
    else:
      # loads last checkpoint, unless `reset` is true
      model.initialize(sess, config.checkpoints, reset=args.reset, reset_learning_rate=args.reset_learning_rate)

    # Inspect variables:
    # tf.get_variable_scope().reuse_variables()
    # import pdb; pdb.set_trace()

    if args.decode:
      model.decode(sess, **config)
    elif args.eval:
      model.evaluate(sess, on_dev=False, **config)
    elif args.train:
      eval_output = os.path.join(config.model_dir, 'eval')
      try:
        model.train(sess, eval_output=eval_output, **config)
      except KeyboardInterrupt:
        utils.log('exiting...')
        model.save(sess)
        sys.exit()

if __name__ == "__main__":
  main()
