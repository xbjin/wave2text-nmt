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

from multi_encoder import data_utils
from multi_encoder import seq2seq_model

from multi_encoder import utils


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Initial learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decay factor")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
tf.app.flags.DEFINE_integer("batch_size", 64, "Training batch size")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each layer")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model")
tf.app.flags.DEFINE_integer("src_vocab_size", 30000, "Source vocabulary size")
tf.app.flags.DEFINE_integer("trg_vocab_size", 30000, "Target vocabulary size")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint")
tf.app.flags.DEFINE_integer("steps_per_eval", 4000, "How many training steps to do per BLEU evaluation")

tf.app.flags.DEFINE_string("decode", None, "Translate this corpus")
tf.app.flags.DEFINE_string("eval", None, "Compute BLEU score on this corpus")

tf.app.flags.DEFINE_boolean("download", False, "Download WMT data")
tf.app.flags.DEFINE_boolean("tokenize", False, "Tokenize data on the fly")
tf.app.flags.DEFINE_integer("gpu_id", None, "Index of the GPU where to run the computation (default: 0)")
tf.app.flags.DEFINE_boolean("no_gpu", False, "Train model on CPU")
tf.app.flags.DEFINE_boolean("reset", False, "Reset model (don't load any checkpoint)")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")

tf.app.flags.DEFINE_string("train_corpus", "train", "Name of the training corpus")
tf.app.flags.DEFINE_string("dev_corpus", "dev", "Name of the development corpus")
tf.app.flags.DEFINE_string("embedding", None, "Name of the embedding files")
tf.app.flags.DEFINE_string("src_ext", "en", "Source file extension(s) (comma-separated)")
tf.app.flags.DEFINE_string("trg_ext", "fr", "Target file extension")

tf.app.flags.DEFINE_string("bleu_script", "scripts/multi-bleu.perl", "Path to BLEU script")
tf.app.flags.DEFINE_boolean("pretrain", False, "Toggle pre-training")
tf.app.flags.DEFINE_string("encoder_num", None, "List of encoder ids to include in the model (comma-separated)")
tf.app.flags.DEFINE_string("model_name", None, "Name of the model")
tf.app.flags.DEFINE_string("embedding_train", None, "List of True/False according to the embedding and src_ext parameter")
tf.app.flags.DEFINE_boolean("dropout_rate", 0, "Dropout rate applied to the LSTM units")

FLAGS = tf.app.flags.FLAGS

data_utils.extract_filenames(FLAGS)  # add filenames to namespace
data_utils.extract_embedding(FLAGS)  # add embeddings to namespace

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# TODO: pick bucket sizes automatically
# The same bucket size is used for all encoders.
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
        if (len(target_ids) < target_size and
            all(len(ids_) < source_size for ids_ in source_ids)):
          data_set[bucket_id].append(source_ids + [target_ids])
          break

  return data_set


def create_model(session, forward_only, reuse=None, model_name=None,
                 initialize=True, embeddings=None, encoder_count=None,
                 encoder_num=None):
  """Create translation model and initialize or load parameters in session."""

  device = None
  if FLAGS.no_gpu:
    device = '/cpu:0'
  elif FLAGS.gpu_id is not None:
    device = '/gpu:{}'.format(FLAGS.gpu_id)

  encoder_count = encoder_count or (FLAGS.src_ext.count(',') + 1)
  encoder_num = encoder_num or (FLAGS.encoder_num.split(',')
                                if FLAGS.encoder_num else None)
  embeddings = embeddings or FLAGS.embeddings

  logging.info('Using device: {}'.format(device))
  with tf.device(device):
    model = seq2seq_model.Seq2SeqModel(
      FLAGS.src_vocab_size, FLAGS.trg_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, encoder_count=encoder_count,
      reuse=reuse, encoder_num=encoder_num, model_name=model_name,
      embedding=embeddings, dropout_rate=FLAGS.dropout_rate)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if initialize:
      if not FLAGS.reset and ckpt and tf.gfile.Exists(
              ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from {}".format(
          ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
      else:
        logging.info("Created model with fresh parameters")
        session.run(tf.initialize_all_variables())
       
  return model


def train():
  logging.info("Preparing data in {}".format(FLAGS.data_dir))
  data_utils.prepare_data(FLAGS)

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
    model = create_model(sess, False)

    logging.info('Printing variables')
    for e in tf.all_variables():
      logging.info('name={}, shape={}'.format(e.name, e.get_shape()))

    # Read data into buckets and compute their sizes.
    logging.info("Reading development and training data (limit: {})".format(
                 FLAGS.max_train_data_size))

    dev_set = read_data(FLAGS.src_dev_ids, FLAGS.trg_dev_ids)
    train_set = read_data(FLAGS.src_train_ids, FLAGS.trg_train_ids,
                          FLAGS.max_train_data_size)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      r = np.random.random_sample()
      bucket_id = min(i for i in xrange(len(train_buckets_scale))
                      if train_buckets_scale[i] > r)

      # Get a batch and make a training step
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      # average loss over last steps
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Save checkpoint, print stats and evaluate
      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')

        logging.info("global step {} learning rate {:.4f} step-time {:.2f} "
          "perplexity {:.2f}".format(model.global_step.eval(),
            model.learning_rate.eval(), step_time, perplexity))

        # Decrease learning rate if no improvement over the last 3 updates
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)

        previous_losses.append(loss)
        # Save checkpoint
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        # Compute perplexity on dev set
        for bucket_id in xrange(len(_buckets)):
          if not dev_set[bucket_id]:
            logging.info("  eval: empty bucket {}".format(bucket_id))
            continue

          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          logging.info("  eval: bucket {} perplexity {:.2f}".format(
            bucket_id, eval_ppx))
        sys.stdout.flush()

      # BLEU evaluation on full dev set
      if current_step % FLAGS.steps_per_eval == 0:
        logging.info('starting BLEU evaluation')
        input_filenames = [FLAGS.trg_dev] + FLAGS.src_dev
        output_filename = os.path.join(FLAGS.train_dir,
                                            "eval.{}.out".format(current_step))
        decode(sess, model, filenames=input_filenames, output=output_filename,
               evaluate=True)


def decode_sentence(sess, model, src_sentences, src_vocab, rev_trg_vocab):
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
    logging.warn("Line is too long ({} tokens). "
                 "It will be truncated".format(len_))
    token_ids = [ids_[:max_len] for ids_ in token_ids]

  bucket_id = min(b for b in xrange(len(_buckets)) if
                  all(_buckets[b][0] > len(ids_) for ids_ in token_ids))

  # Get a 1-element batch to feed the sentence to the model
  data = [token_ids + [[]]]
  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
    {bucket_id: data}, bucket_id)

  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)

  # Greedy decoder FIXME: no beam-search?
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

  # Remove EOS symbols from output
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]

  return ' '.join(rev_trg_vocab[i] for i in outputs)


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
  model = model or create_model(sess, forward_only=True)

  train_batch_size = model.batch_size
  model.batch_size = 1  # decode one sentence at a time

  # Load vocabulary
  src_vocab = [data_utils.initialize_vocabulary(vocab)[0]
               for vocab in FLAGS.src_vocab]
  _, rev_trg_vocab = data_utils.initialize_vocabulary(FLAGS.trg_vocab)

  # if filenames isn't specified, use default files
  if not filenames and evaluate:  # evaluation mode
    filenames = ['{}.{}'.format(FLAGS.eval, ext) for ext in
                 FLAGS.src_ext.split(',')]
  elif not filenames:   # decode mode
    filenames = ['{}.{}'.format(FLAGS.decode, ext) for ext in
                 FLAGS.src_ext.split(',')]

  with utils.open_files(filenames) as files:
    references = None
    if evaluate:  # evaluation mode: first file is the reference
      references = [line.strip() for line in files.pop(0)]

    hypotheses = (
      decode_sentence(sess, model, src_sentences, src_vocab, rev_trg_vocab)
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


def pretrain():
  print("Preparing WMT data in %s" % FLAGS.data_dir)
      
      # limit the amount of memory used to 2/3 of total memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)   
      
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      
    encoder_count = FLAGS.src_ext.count(',') + 1
        
    print("Creating %d encoder(s) with %d layers of %d units." %
          (encoder_count, FLAGS.num_layers, FLAGS.size))

    dummy = create_model(sess, False, reuse=False, model_name="dummy", initialize=False)
    
    # we pretrain, therefore encoder_count is not FLAGS.src_ext.count(',') anymore, its 1
    # if encoder_num specified, we send for each model the num encoder of the flag
    models = [create_model(
             sess, forward_only=False, encoder_count=1, reuse=True,
             encoder_num=FLAGS.encoder_num if FLAGS.encoder_num is None else FLAGS.encoder_num.split(",")[i],
             model_name=FLAGS.model_name.split(",")[i],
             initialize=(i == encoder_count - 1),
             embeddings=[FLAGS.embeddings[i]]+[FLAGS.embeddings[-1]] #send embed of enc + embed of dec
             ) 
             for i in range(encoder_count)]
    
    print ("Reading development and training data (limit: %d)."    
           % FLAGS.max_train_data_size)
    
    #instead of one list of x src vocab (when aligned)
    #we have x list of one src vocab (unaligned pretrain)
    dev_sets = [read_data([FLAGS.src_dev_ids[i]], FLAGS.trg_dev_ids) 
                                      for i in range(encoder_count)]
                                                                  
    train_sets = [read_data([FLAGS.src_train_ids[i]], FLAGS.trg_train_ids,
                    FLAGS.max_train_data_size) for i in range(encoder_count)]

    train_bucket_sizes = [[len(train_sets[i][b]) for b in xrange(len(_buckets))] 
                                        for i in range(encoder_count)]
                                                            
    
    train_total_size = [float(sum(sizes)) for sizes in train_bucket_sizes]
    
    
    train_buckets_scale = [[sum(train_bucket_sizes[i][:b + 1]) / train_total_size[i]
                           for b in xrange(len(train_bucket_sizes[i]))]
                           for i in range(len(train_bucket_sizes))]

    step_times = [0.0 for _ in range(encoder_count)]
    losses = [0.0 for _ in range(encoder_count)]
    previous_losses_s = [[] for _ in range(encoder_count)]
    saver_flag = False
    current_step = 1

    while 1:   
      random_number_01 = np.random.random_sample()
      for i in range(encoder_count):
        bucket_id = min(b for b in xrange(len(train_buckets_scale[i]))
                        if train_buckets_scale[i][b] > random_number_01)

        #update of the model parameters
        model = models[i]
        train_set= train_sets[i]
        dev_set = dev_sets[i]

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)

        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                         target_weights, bucket_id, False)

        step_times[i] += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        losses[i] += step_loss / FLAGS.steps_per_checkpoint

        # params = tf.all_variables()
        # for e in params:
        #   if("EmbeddingWrapper" in e.name):
        #   print(e.name, " " , e.eval(sess))
        # print(e.name)
        # sys.exit(1)

#        params = tf.trainable_variables()
#        
#        for e in params:
#        
#            print(e.name)
#        
#        sys.exit(1)
#        
        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          # Print statistics for the previous epoch.
          perplexity = math.exp(losses[i]) if losses[i] < 300 else float('inf')
          print ("MODEL %s : global step %d learning rate %.4f step-time %.2f perplexity "
                 "%.2f" % (model.model_name, model.global_step.eval(), model.learning_rate.eval(),
                           step_times[i], perplexity))
          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_losses_s[i]) > 2 and losses[i] > max(previous_losses_s[i][-3:]):
            sess.run(model.learning_rate_decay_op)
          previous_losses_s[i].append(losses[i])

          saver_flag = True

          # Run evals on development set and print their perplexity.
          for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                dev_set, bucket_id)
            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          sys.stdout.flush()

      if saver_flag:
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_times = [0.0 for _ in range(encoder_count)]
        losses = [0.0 for _ in range(encoder_count)]
        saver_flag = False
      current_step += 1

        
def main(_):
  if not FLAGS.eval and not FLAGS.decode:  # no logging in decode mode
    logging_level = logging.DEBUG if FLAGS.verbose else logging.INFO
    logging.basicConfig(format='%(message)s', level=logging_level)

  if FLAGS.decode:
    decode()
  elif FLAGS.eval:
    decode(evaluate=True)
  elif FLAGS.pretrain:
    pretrain()  
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
