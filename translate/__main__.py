# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Binary for training translation models and decoding from them.

See the following papers for more information on neural translation models.
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
parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
parser.add_argument('--size', type=int, default=128, help='size of each layer')
parser.add_argument('--num-layers', type=int, default=1, help='number of layers in the model')
parser.add_argument('--src-vocab-size', type=int, nargs='+', default=(30000,), help='source vocabulary size(s)')
parser.add_argument('--trg-vocab-size', type=int, nargs='+', default=(30000,), help='target vocabulary size(s)')
parser.add_argument('--max-train-size', type=int, help='maximum size of training data (default: no limit)')
parser.add_argument('--steps-per-checkpoint', type=int, default=10, help='number of updates per checkpoint')
parser.add_argument('--steps-per-eval', type=int, default=50, help='number of updates per BLEU evaluation')
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
parser.add_argument('--train-prefix', default='train', help='name of the training corpus')
parser.add_argument('--dev-prefix', default='dev', help='name of the development corpus')
parser.add_argument('--embedding-prefix', help='prefix of the embedding files')
parser.add_argument('--checkpoint-prefix', help='prefix of the checkpoint (if --load-checkpoints and --reset are '
                                                'not specified, will try to load earlier versions of this checkpoint')
parser.add_argument('--load-checkpoints', nargs='+', help='list of checkpoints to load (in loading order)')
parser.add_argument('--src-ext', nargs='+', default=('fr',), help='source file extension(s) '
                                                                  '(also used as encoder ids)')
parser.add_argument('--trg-ext', nargs='+', default=('en',), help='target file extension(s) '
                                                                  '(also used as decoder ids)')
parser.add_argument('--bleu-script', default='scripts/multi-bleu.perl', help='path to BLEU script')

# TODO: list of extensions
parser.add_argument('--fixed-embeddings', nargs='+', help='list of extensions for which to fix the embeddings during '
                                                          'training')
parser.add_argument('--log-file', help='log to this file instead of standard output')
parser.add_argument('--replace-unk', help='replace unk symbols in the output (requires special pre-processing)',
                    action='store_true')


def main():
  args = parser.parse_args()
  logging_level = logging.DEBUG if args.verbose else logging.INFO
  logging.basicConfig(filename=args.log_file, format='%(asctime)s %(message)s', level=logging_level,
                      datefmt='%m/%d %H:%M:%S')
  
  logging.info(args)   # TODO: nicer logging
  
  # enforce constraints
  assert len(args.src_ext) == len(args.src_vocab_size)
  assert len(args.trg_ext) == len(args.trg_vocab_size)

  if not os.path.exists(args.train_dir):
    logging.info("Creating directory {}".format(args.train_dir))
    os.makedirs(args.train_dir)

  filenames = utils.get_filenames(**vars(args))
  #embeddings = utils.read_embeddings(**vars(args))
  embeddings = None

  # NMT model parameters
  parameters = namedtuple('parameters', ['dropout_rate', 'max_gradient_norm', 'batch_size', 'size', 'num_layers',
                                         'src_vocab_size', 'trg_vocab_size'])
  parameter_values = parameters(**{k: v for k, v in vars(args).items() if k in parameters._fields})

  checkpoint_prefix = (args.checkpoint_prefix or
                       'checkpoints.{}_{}'.format('-'.join(args.src_ext), '-'.join(args.trg_ext)))
  checkpoint_dir = os.path.join(args.train_dir, checkpoint_prefix)
  checkpoints = args.load_checkpoints and [os.path.join(args.train_dir, checkpoint)
                                           for checkpoint in args.load_checkpoints]
  
  device = None
  if args.no_gpu:
    device = '/cpu:0'
  elif args.gpu_id is not None:
    device = '/gpu:{}'.format(args.gpu_id)  
  
  logging.info('creating model')
  logging.info('using device: {}'.format(device))
  
  with tf.device(device):
    model = TranslationModel(args.src_ext, args.trg_ext, parameter_values, embeddings, checkpoint_dir,
                             args.learning_rate, args.learning_rate_decay_factor, multi_task=args.multi_task)

  config = tf.ConfigProto(log_device_placement=args.verbose, allow_soft_placement=True)
  with tf.Session(config=config) as sess:
    model.initialize(sess, checkpoints, reset=args.reset, reset_learning_rate=args.reset_learning_rate)
    
    if args.decode:
      model.decode(sess, filenames, output=None)
    elif args.eval:
      model.evaluate(sess, filenames, bleu_script=args.bleu_script)
    else:
      try:
        model.train(sess, filenames, args.steps_per_checkpoint, args.steps_per_eval, args.bleu_script,
                    args.max_train_size)
      except KeyboardInterrupt:
        logging.info('exiting...')
        model.save(sess)
        sys.exit()

if __name__ == "__main__":
  main()