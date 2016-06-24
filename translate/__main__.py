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
parser.add_argument('--export-embeddings', nargs='+', help='list of extensions for which to export the embeddings')
parser.add_argument('--debug', action='store_true')

# Model parameters
parser.add_argument('--learning-rate', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--learning-rate-decay-factor', type=float, default=0.95, help='learning rate decay factor')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='clip gradients to this norm')
parser.add_argument('--dropout-rate', type=float, default=0.0, help='dropout rate applied to the LSTM units')
parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
parser.add_argument('--size', type=int, default=1024, help='size of each layer')
parser.add_argument('--embedding-size', type=int, help='size of the embeddings')
parser.add_argument('--num-layers', type=int, default=1, help='number of layers in the model')
parser.add_argument('--vocab-size', type=int, default=30000)
parser.add_argument('--num-samples', type=int, default=512, help='number of samples for sampled softmax (0 for '
                                                                  'standard softmax')
parser.add_argument('--src-vocab-size', type=int, nargs='+', help='source vocabulary size(s) (overrides --vocab-size)')
parser.add_argument('--trg-vocab-size', type=int, nargs='+', help='target vocabulary size(s) (overrides --vocab-size)')
parser.add_argument('--max-train-size', type=int, help='maximum size of training data (default: no limit)')
parser.add_argument('--steps-per-checkpoint', type=int, default=1000, help='number of updates per checkpoint')
parser.add_argument('--steps-per-eval', type=int, default=4000, help='number of updates per BLEU evaluation')
parser.add_argument('--reset-learning-rate', help='reset learning rate (useful for pre-training)', action='store_true')
parser.add_argument('--multi-task', help='train each encoder as a separate task', action='store_true')
parser.add_argument('--task-ratio', type=float, nargs='+', help='ratio of each task')
parser.add_argument('--output', help='output file for decoding')
parser.add_argument('--train-prefix', default='train', help='name of the training corpus')
parser.add_argument('--dev-prefix', default='dev', help='name of the development corpus')

parser.add_argument('--embedding-prefix', default='vectors', help='prefix of the embedding files to use as '
                                                                  'initialization (won\'t be used if the parameters '
                                                                  'are loaded from a checkpoint)')
parser.add_argument('--load-embeddings', nargs='+', help='list of extensions for which to load the embeddings')
parser.add_argument('--checkpoint-prefix', help='prefix of the checkpoint (if --load-checkpoints and --reset are '
                                                'not specified, will try to load earlier versions of this checkpoint')
parser.add_argument('--load-checkpoints', nargs='+', help='list of checkpoints to load (in loading order)')
parser.add_argument('--src-ext', nargs='+', default=['fr',], help='source file extension(s) '
                                                                  '(also used as encoder ids)')
parser.add_argument('--trg-ext', nargs='+', default=['en',], help='target file extension(s) '
                                                                  '(also used as decoder ids)')
parser.add_argument('--bleu-script', default='scripts/multi-bleu.perl', help='path to BLEU script')
parser.add_argument('--fixed-embeddings', nargs='+', help='list of extensions for which to fix the embeddings during '
                                                          'training (deprecated, use --freeze-variables)')
parser.add_argument('--log-file', help='log to this file instead of standard output')
parser.add_argument('--replace-unk', help='replace unk symbols in the output (requires special pre-processing)',
                    action='store_true')
parser.add_argument('--norm-embeddings', help='normalize embeddings', action='store_true')
parser.add_argument('--num-best-checkpoints', type=int, default=5, help='save the x best checkpoints')


# Tensorflow configuration
parser.add_argument('--gpu-id', type=int, default=None, help='index of the GPU where to run the computation')
parser.add_argument('--no-gpu', help='train model on CPU', action='store_true')
parser.add_argument('--mem-fraction', type=float, help='maximum fraction of GPU memory to use', default=1.0)
parser.add_argument('--allow-growth', help='allow GPU memory allocation to change during runtime',
                    action='store_true')
parser.add_argument('--beam-size', type=int, default=4, help='beam size for decoding')
parser.add_argument('--freeze-variables', nargs='+', help='list of variables to freeze during training')
parser.add_argument('--bidir', action='store_true')

"""
data: http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/

Features:
- bi-directional rnn
- keep best checkpoint (in terms of BLEU score)
- try getting rid of buckets (by using dynamic_rnn for encoder + custom dynamic rnn for decoder)
- integrate external features into the decoder (e.g. language model)
- audio features for speech recognition
- model ensembling
- local attention model
- pooling between encoder layers
- use state_is_tuple=True in LSTM
- copy vocab to model dir
- train dir/data dir should be optional
- AdaDelta
- rename scopes to nicer names (+ do mapping of trained models)
- move to tensorflow 0.9

Benchmarks:
- compare our baseline system with vanilla Tensorflow seq2seq
- try replicating Jean et al. (2015)'s results
- analyze the impact of this initial_state_attention parameter (pain in the ass for beam-search decoding)
- test beam-search (beam=1...10) : to be fair, model should be trained with initial_state_attention=True,
  and a single bucket (because our beam-search decoder uses these settings).
- compare beam-search with beam_size=1 with greedy search (they should give the same results)
- try reproducing the experiments of the WMT paper on neural post-editing
- test convolutional attention (on speech recognition?)


Compare results with Jean et al.
Data: WMT14 English->French, news-test-2014 for testing, news-test-2012+2013 for dev
Pre-processing: max size 50, tokenization, no lowercasing, no normalization
Settings: vocab size 30000, GRU units, bi-directional encoder, cell size 1000,
embedding size 620, beam size 12, batch size 80, no softmax sampling, single bucket of size 51
"""


def main(args=None):
  args = parser.parse_args(args)

  if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)

  logging_level = logging.DEBUG if args.verbose else logging.INFO
  logger = utils.create_logger(args.log_file)
  logger.setLevel(logging_level)

  utils.log(' '.join(sys.argv))
  commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
  utils.log('commit hash {}'.format(commit_hash))

  utils.log('program arguments')
  for k, v in vars(args).items():
    utils.log('  {:<20} {}'.format(k, v))

  if args.src_vocab_size is None:
    args.src_vocab_size = [args.vocab_size for _ in args.src_ext]
  if args.trg_vocab_size is None:
    args.trg_vocab_size = [args.vocab_size for _ in args.trg_ext]

  # enforce constraints
  assert len(args.src_ext) == len(args.src_vocab_size), (
    '--src-vocab-size takes {} parameter(s)'.format(len(args.src_ext)))
  assert len(args.trg_ext) == len(args.trg_vocab_size), (
    '--trg-vocab-size takes {} parameter(s)'.format(len(args.trg_ext)))
  assert args.task_ratio is None or len(args.task_ratio) == len(args.src_ext), (
    '--task-ratio takes {} parameter(s)'.format(len(args.src_ext)))
  assert len(set(args.src_ext + args.trg_ext)) == len(args.src_ext + args.trg_ext), (
    'all extensions need to be unique')
  assert args.decode or args.eval or args.export_embeddings or args.train, (
    'you need to specify at least one action (decode, eval, export-embeddings or train)')

  filenames = utils.get_filenames(**vars(args))
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

  embeddings = utils.read_embeddings(filenames, **vars(args))
 
  utils.debug('embeddings {}'.format(embeddings))

  # NMT model parameters
  parameters = namedtuple('parameters', ['dropout_rate', 'max_gradient_norm', 'batch_size', 'size', 'num_layers',
                                         'src_vocab_size', 'trg_vocab_size', 'embedding_size',
                                         'bidir', 'freeze_variables', 'num_samples'])
  parameter_values = parameters(**{k: v for k, v in vars(args).items() if k in parameters._fields})

  checkpoint_prefix = (args.checkpoint_prefix or
                       'checkpoints.{}_{}'.format('-'.join(args.src_ext), '-'.join(args.trg_ext)))
  checkpoint_dir = os.path.join(args.train_dir, checkpoint_prefix)
  checkpoints = args.load_checkpoints and [os.path.join(args.train_dir, checkpoint)
                                           for checkpoint in args.load_checkpoints]
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
                             task_ratio=args.task_ratio, num_best_checkpoints = args.num_best_checkpoints)

  utils.log('model parameters ({})'.format(len(tf.all_variables())))
  for var in tf.all_variables():
    utils.log('  {} shape {}'.format(var.name, var.get_shape()))

  config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  config.gpu_options.allow_growth = args.allow_growth
  config.gpu_options.per_process_gpu_memory_fraction = args.mem_fraction

  with tf.Session(config=config) as sess:
    model.initialize(sess, checkpoints, reset=args.reset, reset_learning_rate=args.reset_learning_rate)

    if args.decode:
      model.decode(sess, filenames, args.beam_size, output=args.output)
    elif args.eval:
      model.evaluate(sess, filenames, args.beam_size, bleu_script=args.bleu_script, output=args.output)
    elif args.export_embeddings:
      model.export_embeddings(sess, filenames, extensions=args.export_embeddings,
                              output_prefix=os.path.join(args.train_dir, args.embedding_prefix))
    elif args.train:
      try:
        model.train(sess, filenames, args.beam_size, args.steps_per_checkpoint, args.steps_per_eval, args.bleu_script,
                    args.max_train_size, eval_output)
      except KeyboardInterrupt:
        utils.log('exiting...')
        model.save(sess)
        sys.exit()

if __name__ == "__main__":
  main()