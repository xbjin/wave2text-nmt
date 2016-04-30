from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import cPickle
import logging
import time
import math
import numpy as np
from translate import seq2seq_model, data_utils


class TranslationModel(object):
  def __init__(self, src_ext, trg_ext, parameters, embeddings, checkpoint_dir, learning_rate,
               learning_rate_decay_factor, multi_task=False):
    self.buckets = [(10, 5), (15, 10), (25, 20), (51, 51)]
    self.checkpoint_dir = checkpoint_dir
    self.multi_task = multi_task    
    
    self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)    
    
    with tf.device('/cpu:0'):
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # main model
    self.model = seq2seq_model.Seq2SeqModel(src_ext, trg_ext, self.buckets, self.learning_rate, self.global_step,
                                            **vars(parameters))    
    self.models = []
    
    if multi_task:  # multi-task
      for ext, vocab_size in zip(src_ext, parameters.src_vocab_size):
        params = {k: v for k, v in vars(parameters).items() if k != 'src_vocab_size'}  # FIXME: ugly
        partial_model = seq2seq_model.Seq2SeqModel([ext], trg_ext, self.buckets, self.learning_rate, self.global_step,
                                                   src_vocab_size=[vocab_size], reuse=True, **params)
        self.models.append(partial_model)
    else:  # multi-source
      self.models.append(self.model)

  def read_data(self, filenames, max_train_size):
    if self.multi_task:
      for model, src_train_ids, trg_train_ids in zip(self.models, filenames.src_train_ids, filenames.trg_train_ids):
        train_set = ([src_train_ids], trg_train_ids)
        model.read_data(train_set, self.buckets, max_train_size=max_train_size)
      
    else:
      train_set = (filenames.src_train_ids, filenames.trg_train_ids)
      self.model.read_data(train_set, self.buckets, max_train_size)
      
    self.model.dev_set = data_utils.read_dataset(filenames.src_dev_ids, filenames.trg_dev_ids, self.buckets)

  def initialize(self, sess, checkpoints=None, reset=False, reset_learning_rate=False):
    sess.run(tf.initialize_all_variables())
    if not reset:
      blacklist = ('learning_rate',) if reset_learning_rate else ()
      load_checkpoint(sess, self.checkpoint_dir, blacklist=blacklist)
      
    if checkpoints is not None:  # load partial checkpoints
      for checkpoint in checkpoints:
        load_checkpoint(sess, checkpoint, blacklist=('learning_rate', 'global_step'))

  def train(self, sess, steps_per_checkpoint, steps_per_eval=None):
    previous_losses = []
      
    losses = [0.0] * len(self.models)
    times = [0.0] * len(self.models)
    steps = [0] * len(self.models)
    
    while True:
      i = np.random.randint(len(self.models))
      model = self.models[i]

      start_time = time.time()
      step_loss = self.train_step(sess, model)

      times[i] += (time.time() - start_time)
      losses[i] += step_loss
      steps[i] += 1
      
      global_step = self.global_step.eval(sess)
      
      if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:        
        loss = sum(losses) / steps_per_checkpoint
        step_time = sum(times) / steps_per_checkpoint
        perplexity = math.exp(loss) if loss < 300 else float('inf')

        logging.info('global step {} learning rate {:.4f} step-time {:.2f} perplexity {:.2f}'.format(
          global_step, self.model.learning_rate.eval(), step_time, perplexity))
          
        # decay learning rate when loss is worse than 3 last losses
        if len(previous_losses) > 2 and loss > max(previous_losses):
          sess.run(self.learning_rate_decay_op)

        previous_losses.append(loss)          
        
        if self.multi_task:  # detail per model
          perplexities = [math.exp(loss_ / steps_) if steps_ > 0 and loss_ / steps_ < 300 else float('inf')
                          for loss_, steps_ in zip(losses, steps)]
          step_times = [time_ / steps_ if steps_ > 0 else 0.0 for time_, steps_ in zip(times, steps)]                      
          detail = ' '.join('{{steps {} step-time {:.2f} perplexity {:.2f}}}'.format(steps_, step_time_, perplexity_)
              for steps_, step_time_, perplexity_ in zip(steps, step_times, perplexities))
          logging.info('details per model {}'.format(detail))
        
        losses = [0.0] * len(self.models)
        times = [0.0] * len(self.models)
        steps = [0] * len(self.models)      
        
        self.eval_step(sess, self.model)
        #self.save(sess)
        
      if steps_per_eval and global_step % steps_per_eval == 0:
        logging.info('starting BLEU eval')

  def train_step(self, sess, model):
      r = np.random.random_sample()
      bucket_id = min(i for i in xrange(len(model.train_bucket_scales)) if model.train_bucket_scales[i] > r)          
    
      # get a batch and make a training step
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(model.train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)
      return step_loss

  def eval_step(self, sess, model):
    # compute perplexity on dev set
    for bucket_id in xrange(len(self.buckets)):
      if not model.dev_set[bucket_id]:
        logging.info("  eval: empty bucket {}".format(bucket_id))
        continue

      encoder_inputs, decoder_inputs, target_weights = model.get_batch(model.dev_set, bucket_id)
      _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                   forward_only=True, decode=False)

      perplexity = math.exp(eval_loss) if eval_loss < 300 else float('inf')
      logging.info("  eval: bucket {} perplexity {:.2f}".format(bucket_id, perplexity))

  def decode(self, sess, sentence):
    pass

  def evaluate(self, sess):
    pass

  def save(self, sess):
    save_checkpoint(sess, self.checkpoint_dir, self.global_step)


def load_checkpoint(sess, checkpoint_dir, blacklist=()):
  """ `checkpoint_dir` should be unique to this model """
  var_file = os.path.join(checkpoint_dir, 'vars.pkl')
  
  if os.path.exists(var_file):
    with open(var_file, 'rb') as f:
      var_names = cPickle.load(f)    
      variables = [var for var in tf.all_variables() if var.name in var_names]
  else:
    variables = tf.all_variables()
  
  # remove variables from blacklist
  variables = [var for var in variables if not any(var.name.startswith(prefix) for prefix in blacklist)]
  
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    logging.info('reading model parameters from {}'.format(ckpt.model_checkpoint_path))
    tf.train.Saver(variables).restore(sess, ckpt.model_checkpoint_path)  


def save_checkpoint(sess, checkpoint_dir, step=None):
  """ `checkpoint_dir` should be unique to this model """
  var_file = os.path.join(checkpoint_dir, 'vars.pkl')
  
  if not os.path.exists(checkpoint_dir):
    logging.info("creating directory {}".format(checkpoint_dir))
    os.makedirs(checkpoint_dir)  
  
  with open(var_file, 'wb') as f:
    var_names = [var.name for var in tf.all_variables()]
    cPickle.dump(var_names, f)
  
  logging.info('saving model to {}'.format(checkpoint_dir))
  checkpoint_path = os.path.join(checkpoint_dir, 'translate')
  tf.train.Saver().save(sess, checkpoint_path, step)
  logging.info('finished saving model')
