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

from translate import utils
from collections import namedtuple
from translate.translation_model import TranslationModel


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
parser.add_argument('--reset', help='reset model (don\'t load any checkpoint)', action='store_true')
parser.add_argument('data_dir', default='data', help='data directory')
parser.add_argument('train_dir', default='model', help='training directory')

# Available actions (exclusive)
parser.add_argument('--decode', help='translate this corpus')
parser.add_argument('--eval', help='compute BLEU score on this corpus')
parser.add_argument('--train', help='train an NMT model', action='store_true')

# Model parameters
parser.add_argument('--debug', action='store_true', help='toy settings for debugging (overrides the program '
                                                         'parameters)')
parser.add_argument('--learning-rate', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--learning-rate-decay-factor', type=float, default=0.99, help='learning rate decay factor')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='clip gradients to this norm')
parser.add_argument('--dropout-rate', type=float, default=0.0, help='dropout rate applied to the LSTM units')
parser.add_argument('--no-attention', action='store_true', help='disable the attention mechanism')
parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
parser.add_argument('--size', type=int, default=1024, help='size of each layer')
# TODO: different embedding size for each encoder + decoder
parser.add_argument('--embedding-size', type=int, help='size of the embeddings')
parser.add_argument('--num-layers', type=int, default=1, help='number of layers in the model')
parser.add_argument('--bidir', action='store_true', help='use bidirectional encoder')
parser.add_argument('--attention-window-size', type=int, default=0, help='size of the attention context window '
                                                                         '(local attention), default value of 0 '
                                                                         'means global attention')
parser.add_argument('--attention-filters', type=int, default=0, help='number of convolution filters in attention '
                                                                     'mechanism')
parser.add_argument('--attention-filter-length', type=int, default=10, help='length of convolution filters')
# TODO: pooling over time with multi-encoder, and non-bidir encoder
parser.add_argument('--pooling-ratios', nargs='+', type=int, help='pooling over time between the layers of the '
                                                                  'encoder: 1 out of every n outputs are kept')
parser.add_argument('--vocab-size', type=int, default=40000)
parser.add_argument('--use-lstm', help='use LSTM cells instead of GRU', action='store_true')
parser.add_argument('--num-samples', type=int, default=512, help='number of samples for sampled softmax (0 for '
                                                                  'standard softmax')
parser.add_argument('--src-vocab-size', type=int, nargs='+', help='source vocabulary size(s) (overrides --vocab-size)')
parser.add_argument('--trg-vocab-size', type=int, help='target vocabulary size (overrides --vocab-size)')
parser.add_argument('--max-train-size', type=int, help='maximum size of training data (default: no limit)')
parser.add_argument('--steps-per-checkpoint', type=int, default=1000, help='number of updates per checkpoint '
                                                                           '(warning: saving can take a while)')
parser.add_argument('--steps-per-eval', type=int, default=4000, help='number of updates per BLEU evaluation')
parser.add_argument('--max-steps', type=int, default=0, help='max number of steps before stopping')
parser.add_argument('--reset-learning-rate', help='reset learning rate (useful for pre-training)', action='store_true')
parser.add_argument('--multi-task', help='train each encoder as a separate task', action='store_true')
parser.add_argument('--task-ratio', type=float, nargs='+', help='ratio of each task')
parser.add_argument('--output', help='output file for decoding')
parser.add_argument('--train-prefix', default='train', help='name of the training corpus')
parser.add_argument('--dev-prefix', default='dev', help='name of the development corpus')

parser.add_argument('--embedding-prefix', default='vectors', help='prefix of the embedding files to use as '
                                                                  'initialization (won\'t be used if the parameters '
                                                                  'are loaded from a checkpoint)')
parser.add_argument('--binary-input', nargs='+', help='list of extensions for which the input file contains '
                                                      'vector features instead of token ids (useful for '
                                                      'speech recognition)')
parser.add_argument('--load-embeddings', nargs='+', help='list of extensions for which to load the embeddings')
parser.add_argument('--checkpoint-prefix', help='prefix of the checkpoint (if --reset is not specified, '
                                                'will try to load earlier versions of this checkpoint')
parser.add_argument('--checkpoints', nargs='+', help='list of checkpoints to load (in loading order)')

parser.add_argument('--ensemble', help='use an ensemble of models while decoding, whose checkpoints '
                                       'are those specified by the --load-checkpoints parameter'
                                       '(this changes the semantic of this parameter)', action='store_true')

parser.add_argument('--src-ext', nargs='+', default=['fr',], help='source file extension(s) '
                                                                  '(also used as encoder ids)')
parser.add_argument('--trg-ext', default='en', help='target file extension '
                                                    '(also used as decoder id)')
parser.add_argument('--bleu-script', default='scripts/multi-bleu.perl', help='path to BLEU script')
parser.add_argument('--log-file', help='log to this file instead of standard output')
parser.add_argument('--replace-unk', help='replace UNK symbols in the output (requires special pre-processing'
                                          'and a lookup dict)', action='store_true')
parser.add_argument('--norm-embeddings', help='normalize embeddings', action='store_true')  # FIXME
parser.add_argument('--keep-best', type=int, default=4, help='keep the n best models')
parser.add_argument('--remove-unk', help='remove UNK symbols from the decoder output', action='store_true')
parser.add_argument('--use-lm', help='use language model', action='store_true')
parser.add_argument('--lm-order', type=int, default=3, help='n-gram order of the language model')
parser.add_argument('--lm-prefix', default='train', help='prefix of the language model file (suffix '
                                                         'is always arpa.trg_ext')
parser.add_argument('--model-weights', type=float, nargs='+', help='weight of each model: ensemble models and '
                                                                   'language model (LM weight is last)')

# Tensorflow configuration
parser.add_argument('--gpu-id', type=int, default=None, help='index of the GPU where to run the computation')
parser.add_argument('--no-gpu', help='train model on CPU', action='store_true')
parser.add_argument('--mem-fraction', type=float, help='maximum fraction of GPU memory to use', default=1.0)
parser.add_argument('--allow-growth', help='allow GPU memory allocation to change during runtime',
                    action='store_true')
parser.add_argument('--beam-size', type=int, default=1, help='beam size for decoding (decoder is greedy by default)')
parser.add_argument('--freeze-variables', nargs='+', help='list of variables to freeze during training')
parser.add_argument('--character-level', nargs='+', help='list of extensions whose input is at the character level')
parser.add_argument('--buckets', nargs='+', type=int, help='list of bucket sizes')

"""
data: http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/

Features:
- try getting rid of buckets (by using dynamic_rnn for encoder + custom dynamic rnn for decoder)
- local attention model
- copy vocab to model dir
- train dir/data dir should be optional
- AdaDelta, AdaGrad
- rename scopes to nicer names + do mapping of existing models
- move to tensorflow 0.9

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

  if args.debug:   # toy settings
    args.vocab_size = 10000
    args.size = 128
    args.steps_per_checkpoint = 50
    args.steps_per_eval = 200
    args.verbose = True
    args.batch_size = 32
    args.dev_prefix = 'dev.100'

  if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)

  logging_level = logging.DEBUG if args.verbose else logging.INFO
  logger = utils.create_logger(args.log_file)
  logger.setLevel(logging_level)

  utils.log(' '.join(sys.argv))
  try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    utils.log('commit hash {}'.format(commit_hash))
  except:
    pass

  utils.log('program arguments')
  for k, v in vars(args).items():
    utils.log('  {:<20} {}'.format(k, v))

  extensions = args.src_ext + [args.trg_ext]

  if args.src_vocab_size is None:
    args.src_vocab_size = [args.vocab_size for _ in args.src_ext]
  if args.trg_vocab_size is None:
    args.trg_vocab_size = args.vocab_size

  # enforce constraints
  assert args.steps_per_eval % args.steps_per_checkpoint == 0, (
    'steps-per-eval should be a multiple of steps-per-checkpoint')
  assert len(args.src_ext) == len(args.src_vocab_size), (
    '--src-vocab-size takes {} parameter(s)'.format(len(args.src_ext)))
  assert args.task_ratio is None or len(args.task_ratio) == len(args.src_ext), (
    '--task-ratio takes {} parameter(s)'.format(len(args.src_ext)))
  assert len(set(extensions)) == len(extensions), (
    'all extensions need to be unique')
  assert args.decode or args.eval or args.train, (
    'you need to specify at least one action (decode, eval, or train)')
  assert args.buckets is None or len(args.buckets) % len(extensions) == 0
  assert not (args.use_lm and args.character_level and args.trg_ext in args.character_level), (
    'can\'t use a language model when the output is at the character level')

  filenames = utils.get_filenames(extensions=extensions, **vars(args))
  utils.debug('filenames')
  for k, v in vars(filenames).items():
    utils.log('  {:<20} {}'.format(k, v))

  # flatten list of files
  all_filenames = [filename for names in filenames if names is not None
    for filename in (names if isinstance(names, list) else [names]) if filename is not None]
  all_filenames.append(args.bleu_script)
  # check that those files exist
  for filename in all_filenames:
    if not os.path.exists(filename):
      utils.warn('warning: file {} does not exist'.format(filename))

  vocab_sizes = args.src_vocab_size + [args.trg_vocab_size]
  if args.buckets is not None:
    buckets = [tuple(args.buckets[i:i + len(extensions)])
               for i in range(0, len(args.buckets), len(extensions))]
  else:
    buckets = None
  embeddings = utils.read_embeddings(filenames, extensions, vocab_sizes, **vars(args))
 
  utils.debug('embeddings {}'.format(embeddings))

  # NMT model parameters
  parameters = namedtuple('parameters', ['dropout_rate', 'max_gradient_norm', 'batch_size', 'size', 'num_layers',
                                         'src_vocab_size', 'trg_vocab_size', 'embedding_size',
                                         'bidir', 'freeze_variables', 'num_samples',
                                         'attention_filters', 'attention_filter_length', 'use_lstm',
                                         'pooling_ratios', 'model_weights', 'attention_window_size'])
  parameter_values = parameters(**{k: v for k, v in vars(args).items() if k in parameters._fields})

  checkpoint_prefix = (args.checkpoint_prefix or
                       'checkpoints.{}_{}'.format('-'.join(args.src_ext), args.trg_ext))
  checkpoint_dir = os.path.join(args.train_dir, checkpoint_prefix)
  eval_output = os.path.join(args.train_dir, 'eval.out')
  
  device = None
  if args.no_gpu:
    device = '/cpu:0'
  elif args.gpu_id is not None:
    device = '/gpu:{}'.format(args.gpu_id)
  
  utils.log('creating model')
  utils.log('using device: {}'.format(device))
  
  with tf.device(device):
    model = TranslationModel(args.src_ext, args.trg_ext, parameter_values, embeddings, checkpoint_dir,
                             args.learning_rate, args.learning_rate_decay_factor, multi_task=args.multi_task,
                             task_ratio=args.task_ratio, keep_best=args.keep_best, lm_order=args.lm_order,
                             binary_input=args.binary_input, character_level=args.character_level,
                             buckets=buckets)

    utils.log('model parameters ({})'.format(len(tf.all_variables())))
  for var in tf.all_variables():
    utils.log('  {} shape {}'.format(var.name, var.get_shape()))

  config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  config.gpu_options.allow_growth = args.allow_growth
  config.gpu_options.per_process_gpu_memory_fraction = args.mem_fraction

  with tf.Session(config=config) as sess:
    if args.ensemble and (args.eval or args.decode):
      # create one session for each model in the ensemble
      sess = [tf.Session() for _ in args.checkpoints]
      for sess_, checkpoint in zip(sess, args.checkpoints):
        model.initialize(sess_, [checkpoint], reset=True)
    else:
      model.initialize(sess, args.checkpoints, reset=args.reset, reset_learning_rate=args.reset_learning_rate)

    # TODO: load best checkpoint for eval and decode
    if args.decode:
      model.decode(sess, filenames, args.beam_size, output=args.output, remove_unk=args.remove_unk)
    elif args.eval:
      model.evaluate(sess, filenames, args.beam_size, bleu_script=args.bleu_script, output=args.output,
                     remove_unk=args.remove_unk)
    elif args.train:
      try:
        model.train(sess, filenames, args.beam_size, args.steps_per_checkpoint, args.steps_per_eval, args.bleu_script,
                    args.max_train_size, eval_output, remove_unk=args.remove_unk, max_steps=args.max_steps)
      except KeyboardInterrupt:
        utils.log('exiting...')
        model.save(sess)
        sys.exit()

if __name__ == "__main__":
  main()
