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

import tensorflow as tf

from translate import utils
from collections import namedtuple
from translate.translation_model import TranslationModel


parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--learning-rate-decay-factor', type=float, default=0.99, help='learning rate decay factor')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='clip gradients to this norm')
parser.add_argument('--dropout-rate', type=float, default=0.0, help='dropout rate applied to the LSTM units')
parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
parser.add_argument('--size', type=int, default=1024, help='size of each layer')
parser.add_argument('--num-layers', type=int, default=1, help='number of layers in the model')
parser.add_argument('--src-vocab-size', type=int, nargs='+', default=[30000,], help='source vocabulary size(s)')
parser.add_argument('--trg-vocab-size', type=int, nargs='+', default=[30000,], help='target vocabulary size(s)')
parser.add_argument('--max-train-size', type=int, help='maximum size of training data (default: no limit)')
parser.add_argument('--steps-per-checkpoint', type=int, default=50, help='number of updates per checkpoint')
parser.add_argument('--steps-per-eval', type=int, default=1000, help='number of updates per BLEU evaluation')
parser.add_argument('--gpu-id', type=int, default=None, help='index of the GPU where to run the computation')
parser.add_argument('--no-gpu', help='train model on CPU', action='store_true')
parser.add_argument('--reset', help='reset model (don\'t load any checkpoint)', action='store_true')
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
parser.add_argument('--reset-learning-rate', help='reset learning rate (useful for pre-training)', action='store_true')
parser.add_argument('--multi-task', help='train each encoder as a separate task', action='store_true')
parser.add_argument('data_dir', default='data', help='data directory')
parser.add_argument('train_dir', default='model', help='training directory')
parser.add_argument('--decode', help='translate this corpus')
parser.add_argument('--eval', help='compute BLEU score on this corpus')
parser.add_argument('--output', help='output file for decoding')
parser.add_argument('--train-prefix', default='train', help='name of the training corpus')
parser.add_argument('--dev-prefix', default='dev', help='name of the development corpus')
parser.add_argument('--embedding-prefix', help='prefix of the embedding files to use as initialization (won\'t be used '
                                                'if the parameters are loaded from a checkpoint)')
parser.add_argument('--checkpoint-prefix', help='prefix of the checkpoint (if --load-checkpoints and --reset are '
                                                'not specified, will try to load earlier versions of this checkpoint')
parser.add_argument('--load-checkpoints', nargs='+', help='list of checkpoints to load (in loading order)')
parser.add_argument('--src-ext', nargs='+', default=['fr',], help='source file extension(s) '
                                                                  '(also used as encoder ids)')
parser.add_argument('--trg-ext', nargs='+', default=['en',], help='target file extension(s) '
                                                                  '(also used as decoder ids)')
parser.add_argument('--bleu-script', default='scripts/multi-bleu.perl', help='path to BLEU script')
parser.add_argument('--fixed-embeddings', nargs='+', help='list of extensions for which to fix the embeddings during '
                                                          'training')
parser.add_argument('--log-file', help='log to this file instead of standard output')
parser.add_argument('--replace-unk', help='replace unk symbols in the output (requires special pre-processing)',
                    action='store_true')
# TODO: fixed encoder/decoder


def main():
  args = parser.parse_args()
  logging_level = logging.DEBUG if args.verbose else logging.INFO
  logger = utils.create_logger(args.log_file)
  logger.setLevel(logging_level)
  
  utils.log('program arguments')
  for k, v in vars(args).items():
    utils.log('  {:<20} {}'.format(k, v))
  
  # enforce constraints
  assert len(args.src_ext) == len(args.src_vocab_size)
  assert len(args.trg_ext) == len(args.trg_vocab_size)

  if not os.path.exists(args.train_dir):
    utils.log("Creating directory {}".format(args.train_dir))
    os.makedirs(args.train_dir)

  filenames = utils.get_filenames(**vars(args))
  utils.debug('filenames')
  for k, v in vars(filenames).items():
    utils.log('  {:<20} {}'.format(k, v))
  
  embeddings = utils.read_embeddings(filenames, **vars(args))
  utils.debug('embeddings {}'.format(embeddings))

  # NMT model parameters
  parameters = namedtuple('parameters', ['dropout_rate', 'max_gradient_norm', 'batch_size', 'size', 'num_layers',
                                         'src_vocab_size', 'trg_vocab_size'])
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
                             args.learning_rate, args.learning_rate_decay_factor, multi_task=args.multi_task)

  utils.log('model parameters')
  for var in tf.all_variables():
    utils.log('  {} shape {}'.format(var.name, var.get_shape()))

  config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    model.initialize(sess, checkpoints, reset=args.reset, reset_learning_rate=args.reset_learning_rate)
    
    if args.decode:
      model.decode(sess, filenames, output=args.output)
    elif args.eval:
      model.evaluate(sess, filenames, bleu_script=args.bleu_script, output=args.output)
    else:
      try:
        model.train(sess, filenames, args.steps_per_checkpoint, args.steps_per_eval, args.bleu_script,
                    args.max_train_size, eval_output)
      except KeyboardInterrupt:
        utils.log('exiting...')
        model.save(sess)
        sys.exit()

if __name__ == "__main__":
  main()