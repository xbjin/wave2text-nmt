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

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from translate import data_utils
from translate import seq2seq_model

from translate import utils


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Initial learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decay factor")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
tf.app.flags.DEFINE_float("dropout_rate", 0, "Dropout rate applied to the LSTM units")

tf.app.flags.DEFINE_integer("batch_size", 64, "Training batch size")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each layer")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model")
tf.app.flags.DEFINE_integer("src_vocab_size", 30000, "Source vocabulary size")
tf.app.flags.DEFINE_integer("trg_vocab_size", 30000, "Target vocabulary size")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint")
tf.app.flags.DEFINE_integer("steps_per_eval", 4000, "How many training steps to do per BLEU evaluation")
tf.app.flags.DEFINE_integer("gpu_id", None, "Index of the GPU where to run the computation (default: 0)")

tf.app.flags.DEFINE_boolean("download", False, "Download WMT data")
tf.app.flags.DEFINE_boolean("tokenize", False, "Tokenize data on the fly")
tf.app.flags.DEFINE_boolean("no_gpu", False, "Train model on CPU")
tf.app.flags.DEFINE_boolean("reset", False, "Reset model (don't load any checkpoint)")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")
tf.app.flags.DEFINE_boolean("reset_learning_rate", False, "Reset learning rate (useful for pre-training)")
tf.app.flags.DEFINE_boolean("multi_task", False, "Train each encoder as a separate task")

tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory")
tf.app.flags.DEFINE_string("decode", None, "Translate this corpus")
tf.app.flags.DEFINE_string("eval", None, "Compute BLEU score on this corpus")

tf.app.flags.DEFINE_string("train_prefix", "train", "Name of the training corpus")
tf.app.flags.DEFINE_string("dev_prefix", "dev", "Name of the development corpus")
tf.app.flags.DEFINE_string("embedding_prefix", None, "Prefix of the embedding files")
tf.app.flags.DEFINE_string("src_ext", "fr", "Source file extension(s) (comma-separated)")
tf.app.flags.DEFINE_string("trg_ext", "en", "Target file extension")
tf.app.flags.DEFINE_string("bleu_script", "scripts/multi-bleu.perl", "Path to BLEU script")
tf.app.flags.DEFINE_string("encoder_num", None, "List of comma-separated encoder ids to include in the model "
                                                "(useful for pre-training), same size as src_ext")
tf.app.flags.DEFINE_string("model_name", None, "Name of the model")
tf.app.flags.DEFINE_string("fix_embeddings", None, "List of comma-separated 0/1 values specifying "
                                                   "which embeddings to freeze during training")
tf.app.flags.DEFINE_string("lookup_dict", None, "Dict to replace UNK Tokens")
tf.app.flags.DEFINE_string("logfile", None, "Log to this file instead of standard output")

FLAGS = tf.app.flags.FLAGS

data_utils.extract_filenames(FLAGS)  # add filenames to namespace
data_utils.extract_embedding(FLAGS)  # add embeddings to namespace

# We use a number of buckets and pad to the closest one for efficiency
# See seq2seq_model.Seq2SeqModel for details on how they work
# TODO: pick bucket sizes automatically
# The same bucket size is used for all encoders
_buckets = [(5, 10), (10, 15), (20, 25), (51, 51)]

if FLAGS.trg_ext == 'en':  # temporary hack for fr->en
  _buckets = [tuple(reversed(bucket)) for bucket in _buckets]


def read_data(source_paths, target_path, max_size=None):
  data_set = [[] for _ in _buckets]

  filenames = source_paths + [target_path]
  with utils.open_files(filenames) as files:

    for counter, lines in enumerate(zip(*files), 1):
      if max_size and counter >= max_size:
        break
      if counter % 100000 == 0:
        logging.info("  reading data line {}".format(counter))

      ids = [map(int, line.split()) for line in lines]
      source_ids, target_ids = ids[:-1], ids[-1]

      # FIXME: why only target sequence gets an EOS token?
      target_ids.append(data_utils.EOS_ID)

      if any(len(ids_) == 0 for ids_ in ids):  # skip empty lines
        continue

      for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(target_ids) < target_size and all(len(ids_) < source_size for ids_ in source_ids):
          data_set[bucket_id].append(source_ids + [target_ids])
          break

  return data_set


def create_model(session, reuse=None, model_name=None, initialize=True,
                 embeddings=None, encoder_count=None, encoder_num=None):
  """Create translation model and initialize or load parameters in session."""

  device = None
  if FLAGS.no_gpu:
    device = '/cpu:0'
  elif FLAGS.gpu_id is not None:
    device = '/gpu:{}'.format(FLAGS.gpu_id)

  encoder_count = encoder_count or (FLAGS.src_ext.count(',') + 1)
  encoder_num = encoder_num or (FLAGS.encoder_num.split(',') if FLAGS.encoder_num else None)
  embeddings = embeddings or FLAGS.embeddings

  logging.info('Using device: {}'.format(device))
  with tf.device(device):
    model = seq2seq_model.Seq2SeqModel(
      FLAGS.src_vocab_size, FLAGS.trg_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      encoder_count=encoder_count, reuse=reuse, encoder_num=encoder_num,
      model_name=model_name, embedding=embeddings,
      dropout_rate=FLAGS.dropout_rate)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if initialize:
      if not FLAGS.reset and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
      else:
        logging.info("Created model with fresh parameters")
        session.run(tf.initialize_all_variables())
       
  return model


def train():
  """ Choo-Choo! """

  # print values of program arguments
  arguments = '\n'.join('  {}: {}'.format(k, repr(v)) for k, v in tf.app.flags.FLAGS.__flags.iteritems())
  logging.info('Arguments:\n' + arguments)

  # We assume that data has been prepared with scripts/prepare-data.py
  # logging.info("Preparing data in {}".format(FLAGS.data_dir))
  # data_utils.prepare_data(FLAGS)

  if not os.path.exists(FLAGS.train_dir):
    logging.info("Creating directory {}".format(FLAGS.train_dir))
    os.makedirs(FLAGS.train_dir)

  # limit the amount of memory used to 2/3 of total memory
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
  gpu_options = tf.GPUOptions()
  log_device_placement = FLAGS.verbose

  config = tf.ConfigProto(log_device_placement=log_device_placement,
                          allow_soft_placement=True, gpu_options=gpu_options)

  with tf.Session(config=config) as sess:
    logging.info("Creating {} layers of {} units".format(FLAGS.num_layers,
                                                         FLAGS.size))
    full_model = create_model(sess)   # model that contains all the variables
    if FLAGS.multi_task:  # creating partial models for multi-task training
      logging.info('Multi-task training, creating {} models'.format(FLAGS.encoder_count))
      models = []
      for embeddings, encoder_id in zip(FLAGS.embeddings, FLAGS.encoder_ids):
        model = create_model(sess, encoder_count=1, reuse=True, encoder_num=[encoder_id],
                             embeddings=(embeddings, FLAGS.embeddings[-1]), initialize=False)
        models.append(model)
    else:
      models = [full_model]
    # sess.run(tf.initialize_all_variables())

    logging.info('Printing variables')
    for e in tf.all_variables():
      logging.info('name={}, shape={}'.format(e.name, e.get_shape()))

    # Read data into buckets and compute their sizes.
    max_train_data_size = FLAGS.max_train_data_size

    logging.info("Reading development and training data (limit: {})".format(max_train_data_size))

    dev_set = read_data(FLAGS.src_dev_ids, FLAGS.trg_dev_ids)  # dev set is for the full model

    if FLAGS.multi_task:  # one train set for each partial model
      train_sets = [read_data([src_ids], trg_ids, max_train_data_size)
                    for src_ids, trg_ids in zip(FLAGS.src_train_ids, FLAGS.trg_train_ids)]
    else:
      train_sets = [read_data(FLAGS.src_train_ids, FLAGS.trg_train_ids, max_train_data_size)]

    train_bucket_sizes = [[len(train_set[b]) for b in xrange(len(_buckets))] for train_set in train_sets]
    train_total_sizes = [float(sum(bucket_sizes)) for bucket_sizes in train_bucket_sizes]

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_bucket_scales = [
      [sum(bucket_sizes[:i + 1]) / total_size for i in xrange(len(bucket_sizes))]
      for bucket_sizes, total_size in zip(train_bucket_sizes, train_total_sizes)
    ]

    step_time, loss = 0.0, 0.0
    previous_losses = []

    losses = [0.0] * len(models)
    steps = [0] * len(models)

    while True:
      # randomly choose a model to train
      i = random.randrange(len(models))
      model = models[i]
      bucket_scale = train_bucket_scales[i]
      train_set = train_sets[i]

      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      r = np.random.random_sample()
      bucket_id = min(i for i in xrange(len(bucket_scale)) if bucket_scale[i] > r)

      # Get a batch and make a training step
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      # average loss over last steps
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step = full_model.global_step.eval()

      # update loss and number of steps for selected model
      losses[i] += step_loss
      steps[i] += 1

      # Once in a while, save checkpoint, print stats and evaluate full model
      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')

        logging.info("global step {} learning rate {:.4f} step-time {:.2f} perplexity {:.2f}".format(
          current_step, full_model.learning_rate.eval(), step_time, perplexity))

        if FLAGS.multi_task:  # detail per model
          perplexities = [math.exp(loss / steps_) if loss / steps_ < 300 else float('inf')
                          for loss, steps_ in zip(losses, steps)]

          logging.info('details per model ' + ' '.join('{{steps {} perplexity {}}}'.format(steps_, perplexity)
                       for steps_, perplexity in zip(steps, perplexities)))

        # Decrease learning rate if no improvement over the last 3 updates
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(full_model.learning_rate_decay_op)

        previous_losses.append(loss)
        # Save checkpoint
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        full_model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        losses = [0.0] * len(models)
        steps = [0] * len(models)

        # Compute perplexity on dev set
        for bucket_id in xrange(len(_buckets)):
          if not dev_set[bucket_id]:
            logging.info("  eval: empty bucket {}".format(bucket_id))
            continue

          encoder_inputs, decoder_inputs, target_weights = full_model.get_batch(dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                       forward_only=True, decode=False)

          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          logging.info("  eval: bucket {} perplexity {:.2f}".format(bucket_id, eval_ppx))
        sys.stdout.flush()

      # BLEU evaluation on full dev set
      if current_step % FLAGS.steps_per_eval == 0:
        logging.info('starting BLEU evaluation')
        input_filenames = [FLAGS.trg_dev] + FLAGS.src_dev
        output_filename = os.path.join(FLAGS.train_dir, "eval.out-{}".format(current_step))
        decode(sess, full_model, filenames=input_filenames, output=output_filename, evaluate=True)


def decode_sentence(sess, model, src_sentences, src_vocab, rev_trg_vocab, lookup_dict=None):
  """
  Translate given sentence with the given seq2seq model and
    return the translation.
  """  
  
  tokenizer = data_utils.no_tokenizer if not FLAGS.tokenize else None

  token_ids = [
    data_utils.sentence_to_token_ids(sentence, vocab, tokenizer=tokenizer)
    for sentence, vocab in zip(src_sentences, src_vocab)
  ]
  
  max_len = _buckets[-1][0] - 1
  if any(len(ids_) > max_len for ids_ in token_ids):
    len_ = max(map(len, token_ids))
    logging.warn("Line is too long ({} tokens). It will be truncated".format(len_))
    token_ids = [ids_[:max_len] for ids_ in token_ids]

  bucket_id = min(b for b in xrange(len(_buckets)) if all(_buckets[b][0] > len(ids_) for ids_ in token_ids))

  # Get a 1-element batch to feed the sentence to the model
  data = [token_ids + [[]]]
  encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: data}, bucket_id)

  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                   forward_only=True, decode=True)

  # Greedy decoder FIXME: no beam-search?
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

  # Remove EOS symbols from output
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]

  text_output = [rev_trg_vocab[i] for i in outputs]

  if lookup_dict is not None:
    # first source is used for UNK replacement
    src_tokens = tokenizer(src_sentences[0])
    for trg_pos, trg_id in enumerate(outputs):
      if not 4 <= trg_id <= 19:  # UNK symbols range
        continue

      src_pos = trg_pos + trg_id - 11   # aligned source position (symbol 4 is UNK-7, symbol 19 is UNK+7)
      if 0 <= src_pos < len(src_tokens):
        src_word = src_tokens[src_pos]
        # look for a translation, otherwise take the source word itself (e.g. name or number)
        text_output[trg_pos] = lookup_dict.get(src_word, src_word)
      else:   # aligned position is outside of source sentence, nothing we can do.
        text_output[trg_pos] = '_UNK'

  return ' '.join(text_output)


def decode(sess=None, model=None, filenames=None, output=None, evaluate=False):
  """
  This function has a decoding mode and an evaluation mode, controlled by
    the `evaluate` parameter. Evaluation mode will run on the dev set and
    compute and print a BLEU score. Decoding mode will translate the input
    sentences and output the translations.

  Args:
    sess: session in which to do the decoding, opens a new one if unspecified
    model: seq2seq model to use for decoding, loads a new model if unspecified
    filenames: list of input filenames. In evaluation mode, the first file
      should be the target. If unspecified, dev set is used for evaluation mode,
      value of the `decode` flag for decoding mode.
    output: name of the file where to write the translated sentences.
      If unspecified and in decoding mode, prints to standard output.
    evaluate: toggle evaluation mode
  """

  sess = sess or tf.Session()
  model = model or create_model(sess)
  train_batch_size = model.batch_size
  model.batch_size = 1  # decode one sentence at a time

  # Load vocabulary
  src_vocab = [data_utils.initialize_vocabulary(vocab)[0] for vocab in FLAGS.src_vocab]
  _, rev_trg_vocab = data_utils.initialize_vocabulary(FLAGS.trg_vocab)

  # if filenames isn't specified, use default files
  if not filenames:
    prefix = FLAGS.eval if evaluate else FLAGS.decode
    src_ext = FLAGS.src_ext.split(',')
    extensions = [FLAGS.trg_ext] + src_ext if evaluate else src_ext
    filenames = ['{}.{}'.format(prefix, ext) for ext in extensions]

  lookup_dict = None
  if FLAGS.lookup_dict:
    dict_filename = os.path.join(FLAGS.data_dir, FLAGS.lookup_dict)
    with open(dict_filename) as dict_file:
      lookup_dict = dict(line.split() for line in dict_file)
    
  with utils.open_files(filenames) as files:
    references = None
    if evaluate:  # evaluation mode: first file is the reference
      references = [line.strip() for line in files.pop(0)]

    hypotheses = (
      decode_sentence(sess, model, src_sentences, src_vocab, rev_trg_vocab, lookup_dict)
      for src_sentences in zip(*files)
    )

    if evaluate:  # evaluation mode
      hypotheses = list(hypotheses)
      logging.info('  score {} penalty {} ratio {}'.format(
        *data_utils.bleu_score(FLAGS.bleu_script, hypotheses, references)))

    if output or not evaluate:
      output_file = None
      try:
        output_file = open(output, 'w') if output else sys.stdout
        for line in hypotheses:
          output_file.write(line + '\n')
      finally:
        output_file.close()

  model.batch_size = train_batch_size  # reset batch size to its initial value

        
def main(_):
  if not FLAGS.decode:  # no logging in decoding mode
    logging_level = logging.DEBUG if FLAGS.verbose else logging.INFO
    logging.basicConfig(filename=FLAGS.logfile, format='%(asctime)s %(message)s', level=logging_level,
                        datefmt='%m/%d %H:%M:%S')

  if FLAGS.decode:
    decode()
  elif FLAGS.eval:
    decode(evaluate=True)
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
