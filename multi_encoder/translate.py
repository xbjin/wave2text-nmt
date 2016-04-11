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

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from multi_encoder import data_utils
from multi_encoder import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("src_vocab_size", 40000, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("trg_vocab_size", 40000, "Target vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_eval", 4000, "How many training steps to"
                                                    " do per BLEU evaluation.")
tf.app.flags.DEFINE_string("decode", None, "Corpus to translate.")
tf.app.flags.DEFINE_boolean("eval", False,
                            "Set to True for BLEU evaluation.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("download", False, "Download WMT data.")
tf.app.flags.DEFINE_boolean("tokenize", False, "Tokenize data on the fly.")
tf.app.flags.DEFINE_integer("gpu_id", None, "")
tf.app.flags.DEFINE_boolean("no_gpu", False, "Train model on CPU.")
tf.app.flags.DEFINE_boolean("reset", False, "Reset model "
                                            "(don't load any checkpoint)")

tf.app.flags.DEFINE_string("train_corpus", "train", "Name of the training"
                                                    " corpus.")
tf.app.flags.DEFINE_string("dev_corpus", "dev", "Name of the development"
                                                " corpus.")

tf.app.flags.DEFINE_string("src_ext", "en", "Source files' extension(s),"
                                            "separated by commas")
tf.app.flags.DEFINE_string("trg_ext", "fr", "Target files' extension.")

tf.app.flags.DEFINE_string("bleu_script", "scripts/multi-bleu.perl",
                           "Path to BLEU script.")
tf.app.flags.DEFINE_string("output_file", None, "Output file of the decoder "
                                                "(defaults to stdout).")
tf.app.flags.DEFINE_boolean("pretrain", None, "Wether or not to pretrain")
tf.app.flags.DEFINE_string("encoder_num", None, "List of encoder ids to "
                                                "include in the model, "
                                                "separated by commas ")
tf.app.flags.DEFINE_boolean("create_only", None, "Create the model without "
                                                 "training")
tf.app.flags.DEFINE_string("model_name", None, "Name of the model")

tf.app.flags.DEFINE_string("embedding", None, "Name of the embed files")


FLAGS = tf.app.flags.FLAGS

data_utils.extract_filenames(FLAGS)  # add filenames to namespace
data_utils.extract_embedding(FLAGS) # add embedding to namespace 


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# TODO: pick bucket sizes automatically
# The same bucket size is used for all encoders.
_buckets = [(5, 10), (10, 15), (20, 25), (51, 51)]

if FLAGS.trg_ext == 'en':  # temporary hack for fr->en
  _buckets = [tuple(reversed(bucket)) for bucket in _buckets]


def read_data(source_paths, target_path, max_size=None):
  data_set = [[] for _ in _buckets]

  files = []
  try:
    files = [open(filename) for filename in source_paths + [target_path]]

    for counter, lines in enumerate(zip(*files), 1):
      if max_size and counter >= max_size:
        break
      if counter % 100000 == 0:
        print("  reading data line {}".format(counter))

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

  finally:
    for file_ in files:
      file_.close()

  return data_set


def create_model(session, forward_only, encoder_count, reuse=None,
                 encoder_num=None, model_name=None, initialize=True, embedding=None):
  """Create translation model and initialize or load parameters in session."""

  if FLAGS.no_gpu:
    device = '/cpu:0'
  elif FLAGS.gpu_id is not None:
    device = '/gpu:{}'.format(FLAGS.gpu_id)
  else:
    device = None
    


  print('Using device: {}'.format(device))
  with tf.device(device):
    model = seq2seq_model.Seq2SeqModel(
      FLAGS.src_vocab_size, FLAGS.trg_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, encoder_count=encoder_count,
      device=device, reuse=reuse, encoder_num=encoder_num,
      model_name=model_name, embedding=embedding)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if initialize:
      if not FLAGS.reset and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    
  # if(FLAGS.create_only):
  #   checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
  #   model.saver.save(session, checkpoint_path, global_step=0)
  #   print("Model saved...")
       
  return model


def train():
  print("Preparing WMT data in %s" % FLAGS.data_dir)
  data_utils.prepare_data(FLAGS)

  # limit the amount of memory used to 2/3 of total memory
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
  gpu_options = tf.GPUOptions()
  config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True,
                          gpu_options=gpu_options)

  with tf.Session(config=config) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    encoder_count = FLAGS.src_ext.count(',') + 1
    encoder_num = FLAGS.encoder_num.split(',') if FLAGS.encoder_num else None

    model = create_model(sess, False, encoder_count=encoder_count,
                         encoder_num=encoder_num, embedding=FLAGS.embeddings)

    print('Printing variables')

    for e in tf.all_variables():
      print('name={}, shape={}'.format(e.name, e.get_shape()))

    # Read data into buckets and compute their sizes.
    print("Reading development and training data (limit: %d)."
          % FLAGS.max_train_data_size)

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
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
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


def decode_sentence(sess, model, src_sentences, src_vocabs, rev_trg_vocab=None):
  tokenizer = data_utils.no_tokenizer if not FLAGS.tokenize else None

  token_ids = [
    data_utils.sentence_to_token_ids(sentence, vocab, tokenizer=tokenizer)
    for sentence, vocab in zip(src_sentences, src_vocabs)
  ]

  max_len = _buckets[-1][0] - 1
  if any(len(ids_) > max_len for ids_ in token_ids):
    sys.stderr.write("Line is too long ({} tokens). "
                     "It will be truncated.\n".format(len(token_ids)))
    token_ids = [ids_[:max_len] for ids_ in token_ids]

  bucket_id = min(b for b in xrange(len(_buckets)) if
                  all(_buckets[b][0] > len(ids_) for ids_ in token_ids))

  # Get a 1-element batch to feed the sentence to the model.
  data = [token_ids + [[]]]
  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
    {bucket_id: data}, bucket_id)

  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)

  # This is a greedy decoder - outputs are just argmaxes of output_logits.
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]  # FIXME: no beam-search?

  # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]

  if rev_trg_vocab is not None:
    outputs = ' '.join(rev_trg_vocab[i] for i in outputs)

  return outputs


def decode():
  filenames = ['{}.{}'.format(FLAGS.decode, ext)
               for ext in FLAGS.src_ext.split(',')]

  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    src_vocabs = [data_utils.initialize_vocabulary(vocab)[0]
                  for vocab in FLAGS.src_vocab]

    _, rev_trg_vocab = data_utils.initialize_vocabulary(FLAGS.trg_vocab)

    files = []
    try:
      src_files = [open(filename) for filename in filenames]
      files += src_files

      if FLAGS.output_file:
        output_file = open(FLAGS.output_file, 'w')
        files.append(output_file)
      else:
        output_file = sys.stdout

      # Decode from standard input.
      for i, src_sentences in enumerate(zip(*src_files), 1):
        output = decode_sentence(sess, model, src_sentences, src_vocabs,
                                 rev_trg_vocab)
        output_file.write(output + '\n')

    finally:
      for file_ in files:
        file_.close()


def evaluate():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    filename = FLAGS.output_file
    if filename is None:
      step = model.global_step.eval(session=sess)
      filename = os.path.join(FLAGS.train_dir, 'eval.out-{}'.format(step))

    # Load vocabularies.
    src_vocabs = [data_utils.initialize_vocabulary(vocab)[0]
                  for vocab in FLAGS.src_vocab]

    _, rev_trg_vocab = data_utils.initialize_vocabulary(FLAGS.trg_vocab)

    files = []
    try:
      trg_file = open(FLAGS.trg_dev)
      src_files = [open(filename) for filename in FLAGS.src_dev]
      files += [trg_file] + src_files

      hypotheses = []
      for src_sentences in zip(*src_files):
        hypothesis = decode_sentence(sess, model, src_sentences, src_vocabs, rev_trg_vocab)
        print(hypothesis)
        hypotheses.append(hypothesis)

      references = [line.strip() for line in trg_file]

      with open(filename, 'w') as f:
        f.writelines(line + '\n' for line in hypotheses)

      print(data_utils.bleu_score(FLAGS.bleu_script, hypotheses, references))

    finally:
      for file_ in files:
        file_.close()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def pretrain():
  print("Preparing WMT data in %s" % FLAGS.data_dir)
      
      # limit the amount of memory used to 2/3 of total memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)   
      
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      
    encoder_count = FLAGS.src_ext.count(',') + 1
        
    print("Creating %d encoder(s) with %d layers of %d units." % (encoder_count, FLAGS.num_layers, FLAGS.size))   

    #dummy
    dummy = create_model(sess, False, reuse=False, encoder_count=encoder_count, encoder_num=FLAGS.encoder_num 
            if FLAGS.encoder_num is None else FLAGS.encoder_num.split(","), model_name="dummy", initialize=False,
            embedding=FLAGS.embeddings)
    
    #we pretrain, therefore encoder_count is not FLAGS.src_ext.count(',') anymore, its 1
    #if encoder_num specified, we send for each model the num encoder of the flag
    models = [create_model(
             sess, forward_only=False, encoder_count=1, reuse=True,
             encoder_num=FLAGS.encoder_num if FLAGS.encoder_num is None else FLAGS.encoder_num.split(",")[i],
             model_name=FLAGS.model_name.split(",")[i],
             initialize=(i == encoder_count - 1),
             embedding = [FLAGS.embeddings[i]]
             ) 
             for i in range(encoder_count)]
    
    print ("Reading development and training data (limit: %d)."    
           % FLAGS.max_train_data_size)
    
    #instead of one list of x src vocab (when aligned)
    #we have x list of one src vocab (unaligned pretrain)
    dev_sets = [read_data([FLAGS.src_dev_ids[i]], FLAGS.trg_dev_ids) 
                                                        for i in range(encoder_count)]
                                                                  
    train_sets = [read_data([FLAGS.src_train_ids[i]], FLAGS.trg_train_ids, FLAGS.max_train_data_size) 
                                                        for i in range(encoder_count)]                 


    train_bucket_sizes = [[len(train_sets[i][b]) for b in xrange(len(_buckets))] 
                                                        for i in range(encoder_count)]
                                                            
    
    train_total_size = [float(sum(train_bucket_sizes[i])) for i in range(len(train_bucket_sizes))]
    
    
    train_buckets_scale = [[sum(train_bucket_sizes[i][:b + 1]) / train_total_size[i]
                           for b in xrange(len(train_bucket_sizes[i]))] for i in range(len(train_bucket_sizes))]
                               
    
    step_times = [0.0 for i in range(encoder_count)]
    losses = [0.0 for i in range(encoder_count)]
    previous_losses_s = [[] for i in range(encoder_count)]
    saver_flag = False
    current_step = 1

    while 1:   
        random_number_01 = np.random.random_sample()
        for i in range(encoder_count):           
            bucket_id = min([b for b in xrange(len(train_buckets_scale[i]))
                           if train_buckets_scale[i][b] > random_number_01]) 
            
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
        
            
#            params = tf.all_variables()    
#            for e in params:    
##                if("EmbeddingWrapper" in e.name):
##                    print(e.name, " " , e.eval(sess))
#                print(e.name)
#            sys.exit(1)
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
        
        if(saver_flag):
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_times = [0.0 for i in range(encoder_count)]
            losses = [0.0 for i in range(encoder_count)]     
            saver_flag = False
        current_step += 1

        
def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  elif FLAGS.eval:
    evaluate()
  elif FLAGS.pretrain:
    pretrain()  
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
