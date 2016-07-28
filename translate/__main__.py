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

from operator import itemgetter
from translate import utils
from collections import namedtuple
from translate.translation_model import TranslationModel


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


"""
data: http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/

Features:
- try getting rid of buckets (by using dynamic_rnn for encoder + custom dynamic rnn for decoder)
- copy vocab to model dir
- train dir/data dir should be optional
- AdaDelta, AdaGrad
- rename scopes to nicer names + do mapping of existing models
- move to tensorflow 0.9
- time pooling: concat or sum instead of skipping

Benchmarks:
- compare our baseline system with vanilla Tensorflow seq2seq, and GroundHog/blocks-examples
- try replicating Jean et al. (2015)'s results
- analyze the impact of this initial_state_attention parameter
- try reproducing the experiments of the WMT paper on neural post-editing
- test convolutional attention (on speech recognition)

Evaluation:
scripts/scoring/score.rb --ref {ref} --hyp-detok {hyp} --print
java -jar scripts/meteor-1.5.jar {hyp} {ref} -l {trg_ext} -a ~servan/Tools/METEOR/data/paraphrase-en.gz

BTEC baseline configuration:
python2 -m translate data/btec/ models/btec --size 256 --vocab-size 10000 \
--num-layers 2 --dropout-rate 0.5 --steps-per-checkpoint 1000 --steps-per-eval 2000 \
--learning-rate-decay-factor 0.95 --use-lstm --bidir -v --train --gpu-id 0 --allow-growth \
--src-ext fr --trg-ext en --log-file models/btec/log.txt
"""


def main(args=None):
  args = parser.parse_args(args)

  with open('translate/config/default.yaml') as f:
    default_config = utils.AttrDict(yaml.safe_load(f))

  with open(args.config) as f:
    config = utils.AttrDict(yaml.safe_load(f))
    # command-line parameters have higher precedence than config file
    for k, v in vars(args).items():
      if k in default_config:
        config[k] = v

    # set default values for parameters that are not defined
    for k, v in default_config.items():
      config.setdefault(k, v)

    # AttrDict: easier access to elements (as attributes)
    config.encoders = [utils.AttrDict(encoder) for encoder in config.encoders]
    config.decoder = utils.AttrDict(config.decoder)

  if not os.path.exists(config.model_dir):
    os.makedirs(config.model_dir)

  logging_level = logging.DEBUG if args.verbose else logging.INFO
  logger = utils.create_logger(config.log_file)
  logger.setLevel(logging_level)
  # TODO: copy config file to model dir

  utils.log(' '.join(sys.argv))  # print command line
  try:                           # print git hash
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    utils.log('commit hash {}'.format(commit_hash))
  except:
    pass

  encoders = config.encoders
  decoder = config.decoder
  extensions = [encoder.name for encoder in encoders] + [decoder.name]

  # list of parameters that can be unique to each encoder and decoder
  model_parameters = [
    'cell_size', 'layers', 'vocab_size', 'embedding_size', 'attention_filters', 'attention_filter_length',
    'use_lstm', 'time_pooling', 'attention_window_size', 'dynamic', 'binary', 'character_level', 'bidir'
  ]

  for encoder_or_decoder in encoders + [decoder]:
    for parameter in model_parameters:
      encoder_or_decoder.setdefault(parameter, config.get(parameter))

  utils.log('program arguments')
  for k, v in sorted(config.items(), key=itemgetter(0)):
    if k not in model_parameters:
      utils.log('  {:<20} {}'.format(k, v))

  # enforce constraints
  assert config.steps_per_eval % config.steps_per_checkpoint == 0, (
    'steps-per-eval should be a multiple of steps-per-checkpoint')
  assert args.decode or args.eval or args.train, (
    'you need to specify at least one action (decode, eval, or train)')

  filenames = utils.get_filenames(extensions=extensions, **config)
  utils.debug('filenames')
  for k, v in vars(filenames).items():
    utils.log('  {:<20} {}'.format(k, v))

  # flatten list of files
  all_filenames = [filename for names in filenames if names is not None
    for filename in (names if isinstance(names, list) else [names]) if filename is not None]

  filenames_ = sum([names if isinstance(names, list) else [names] for names in filenames if names is not None], [])
  filenames_.append(config.bleu_script)
  # check that those files exist
  for filename in filenames_:
    if not os.path.exists(filename):
      utils.warn('warning: file {} does not exist'.format(filename))

  # TODO
  # embeddings = utils.read_embeddings(filenames, args.ext, args.vocab_size, **vars(args))
  # utils.debug('embeddings {}'.format(embeddings))

  # TODO: improve checkpoints
  checkpoint_prefix = (config.checkpoint_prefix or
                       'checkpoints.{}_{}'.format('-'.join(extensions[:-1]), extensions[-1]))
  checkpoint_dir = os.path.join(config.model_dir, checkpoint_prefix)
  eval_output = os.path.join(config.model_dir, 'eval.out')
  
  device = None
  if args.no_gpu:
    device = '/cpu:0'
  elif args.gpu_id is not None:
    device = '/gpu:{}'.format(args.gpu_id)
  
  utils.log('creating model')
  utils.log('using device: {}'.format(device))

  with tf.device(device):
    model = TranslationModel(checkpoint_dir=checkpoint_dir, **config)

    utils.log('model parameters ({})'.format(len(tf.all_variables())))
  for var in tf.all_variables():
    utils.log('  {} shape {}'.format(var.name, var.get_shape()))

  tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  tf_config.gpu_options.allow_growth = config.allow_growth
  tf_config.gpu_options.per_process_gpu_memory_fraction = config.mem_fraction

  with tf.Session(config=tf_config) as sess:
    if config.ensemble and (args.eval or args.decode):
      # create one session for each model in the ensemble
      sess = [tf.Session() for _ in config.checkpoints]
      for sess_, checkpoint in zip(sess, config.checkpoints):
        model.initialize(sess_, [checkpoint], reset=True)
    else:
      model.initialize(sess, config.checkpoints, reset=args.reset, reset_learning_rate=args.reset_learning_rate)

    # TODO: load best checkpoint for eval and decode
    if args.decode:
      model.decode(sess, filenames, config.beam_size, output=config.output, remove_unk=config.remove_unk)
    elif args.eval:
      model.evaluate(sess, filenames, config.beam_size, bleu_script=config.bleu_script, output=config.output,
                     remove_unk=config.remove_unk)
    elif args.train:
      try:
        model.train(sess, filenames, config.beam_size, config.steps_per_checkpoint, config.steps_per_eval,
                    config.bleu_script, config.max_train_size, eval_output, remove_unk=config.remove_unk,
                    max_steps=config.max_steps)
      except KeyboardInterrupt:
        utils.log('exiting...')
        model.save(sess)
        sys.exit()

if __name__ == "__main__":
  main()
