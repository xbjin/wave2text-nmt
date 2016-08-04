from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import cPickle
import time
import sys
import math
import numpy as np
import shutil
from translate import utils
from translate.seq2seq_model import Seq2SeqModel
from collections import OrderedDict


class TranslationModel(object):
  def __init__(self, encoders, decoder, checkpoint_dir, learning_rate,
               learning_rate_decay_factor, keep_best=1, buckets=None, **kwargs):
    self.src_ext = [encoder.name for encoder in encoders]
    self.trg_ext = decoder.name
    self.extensions = self.src_ext + [self.trg_ext]

    self.buckets = zip(*([encoder.buckets for encoder in encoders] + [decoder.buckets]))

    self.checkpoint_dir = checkpoint_dir
    self.keep_best = keep_best

    # list of extensions that use vector features instead of text features
    self.binary_input = [encoder.name for encoder in encoders if encoder.binary]
    self.character_level = [encoder.name for encoder in encoders if encoder.character_level]
    
    self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)    
    
    with tf.device('/cpu:0'):
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # main model
    self.model = Seq2SeqModel(encoders, decoder, self.buckets, self.learning_rate, self.global_step, **kwargs)
    self.vocabs = None

    self.saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)

  def _read_data(self, filenames, max_train_size):
    utils.debug('reading vocabularies')
    self._read_vocab(filenames)

    utils.debug('reading training data')
    train_set = utils.read_dataset(filenames.train, self.extensions, self.vocabs, self.buckets,
                                   max_size=max_train_size, binary_input=self.binary_input,
                                   character_level=self.character_level)
    self.model.assign_data_set(train_set)
    
    utils.debug('reading development data')
    dev_set = utils.read_dataset(filenames.dev, self.extensions, self.vocabs, self.buckets,
                                 binary_input=self.binary_input, character_level=self.character_level)
    self.model.dev_set = dev_set

  def _read_vocab(self, filenames):
    if self.vocabs is not None:
      return

    # don't try reading vocabulary for encoders that take pre-computed features
    self.vocabs = [
      utils.initialize_vocabulary(vocab_path) if ext not in self.binary_input else None
      for ext, vocab_path in zip(self.extensions, filenames.vocab)
    ]
    self.src_vocab = self.vocabs[:-1]
    self.trg_vocab = self.vocabs[-1]
    self.ngrams = filenames.lm_path and utils.read_ngrams(filenames.lm_path, self.trg_vocab.vocab)

  def train(self, sess, filenames, beam_size, steps_per_checkpoint, steps_per_eval=None, bleu_script=None,
            max_train_size=None, eval_output=None, remove_unk=False, max_steps=0):
    utils.log('reading training and development data')
    self._read_data(filenames, max_train_size)
    previous_losses = []
      
    loss, time_, steps = 0, 0, 0
    
    utils.log('starting training')
    while True:
      model = self.model

      start_time = time.time()
      step_loss = self._train_step(sess, model)

      time_  += (time.time() - start_time)
      loss += step_loss
      steps += 1
      
      global_step = self.global_step.eval(sess)
      
      if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
        loss_ = loss / steps
        perplexity = math.exp(loss_) if loss_ < 300 else float('inf')

        utils.log('global step {} learning rate {:.4f} step-time {:.2f} perplexity {:.2f}'.format(
          global_step, self.model.learning_rate.eval(), time_ / steps, perplexity))
          
        # decay learning rate when loss is worse than last losses
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          utils.debug('decreasing learning rate')
          sess.run(self.learning_rate_decay_op)

        previous_losses.append(loss)
        loss, time_, steps = 0, 0, 0
        
        self._eval_step(sess, self.model)
        self.save(sess)
        
      if steps_per_eval and bleu_script and global_step % steps_per_eval == 0:
        utils.log('starting BLEU eval')
        output = '{}.{}'.format(eval_output, global_step)
        score = self.evaluate(sess, filenames, beam_size, bleu_script, on_dev=True, output=output,
                              remove_unk=remove_unk)
        self._manage_best_checkpoints(global_step, score)

      if 0 < max_steps < global_step:
        utils.log('finished training')
        return

  def _train_step(self, sess, model):
    r = np.random.random_sample()
    bucket_id = min(i for i in xrange(len(model.train_bucket_scales)) if model.train_bucket_scales[i] > r)
    return model.step(sess, model.train_set, bucket_id)

  def _eval_step(self, sess, model):
    # compute perplexity on dev set
    for bucket_id in xrange(len(self.buckets)):
      if not model.dev_set[bucket_id]:
        utils.log("  eval: empty bucket {}".format(bucket_id))
        continue

      eval_loss = model.step(sess, model.dev_set, bucket_id, forward_only=True)
      perplexity = math.exp(eval_loss) if eval_loss < 300 else float('inf')
      utils.log("  eval: bucket {} perplexity {:.2f}".format(bucket_id, perplexity))

  def _decode_sentence(self, sess, src_sentences, beam_size=1, remove_unk=False):
    token_ids = [utils.sentence_to_token_ids(sentence, vocab.vocab)
                 if vocab is not None else sentence   # when `sentence` is not a sentence but a vector...
                 for vocab, sentence in zip(self.vocabs, src_sentences)]

    for ids_, max_len in zip(token_ids, self.buckets[-1][:-1]):
      if len(ids_) > max_len:
        utils.warn("line is too long ({} tokens), truncating".format(len(ids_)))
        ids_[:] = ids_[:max_len]

    if beam_size <= 1 and not isinstance(sess, list):
      trg_token_ids = self.model.greedy_decoding(sess, token_ids)
    else:
      hypotheses, scores = self.model.beam_search_decoding(sess, token_ids, beam_size, ngrams=self.ngrams,
                                                           reverse_vocab=self.trg_vocab.reverse)
      trg_token_ids = hypotheses[0]   # first hypothesis is the highest scoring one

    # remove EOS symbols from output
    if utils.EOS_ID in trg_token_ids:
      trg_token_ids = trg_token_ids[:trg_token_ids.index(utils.EOS_ID)]

    trg_tokens = [self.trg_vocab.reverse[i] if i < len(self.trg_vocab.reverse) else utils._UNK
                  for i in trg_token_ids]

    if remove_unk:
      trg_tokens = [token for token in trg_tokens if token != utils._UNK]

    if self.trg_ext in self.character_level:
      return ''.join(trg_tokens)
    else:
      return ' '.join(trg_tokens).replace('@@ ', '')  # merge subword units

  def decode(self, sess, filenames, beam_size, output=None, remove_unk=False):
    self._read_vocab(filenames)

    output_file = None
    try:
      output_file = sys.stdout if output is None else open(output, 'w')

      for lines in utils.read_lines(filenames.test[:-1], self.src_ext, self.binary_input):
        trg_sentence = self._decode_sentence(sess, lines, beam_size, remove_unk)
        output_file.write(trg_sentence + '\n')
        output_file.flush()
    finally:
      if output_file is not None:
        output_file.close()

  def evaluate(self, sess, filenames, beam_size, bleu_script, on_dev=False, output=None, remove_unk=False):
    self._read_vocab(filenames)
    if self.ngrams is not None:
      utils.debug('using external language model')

    filenames_ = filenames.dev if on_dev else filenames.test
    lines = list(utils.read_lines(filenames_, self.extensions, self.binary_input))

    hypotheses = [self._decode_sentence(sess, lines_[:-1], beam_size, remove_unk)
                  for lines_ in lines]
    references = [lines_[-1].strip().replace('@@ ', '') for lines_ in lines]

    bleu = utils.bleu_score(bleu_script, hypotheses, references)
    utils.log(bleu)
    if output is not None:
      with open(output, 'w') as f:
        f.writelines(line + '\n' for line in hypotheses)

    return bleu.score

  def _manage_best_checkpoints(self, step, score):
    score_filename = os.path.join(self.checkpoint_dir, 'bleu-scores.txt')
    # try loading previous scores
    try:
      with open(score_filename) as f:
        # list of pairs (score, step)
        scores = [(float(line.split()[0]), int(line.split()[1])) for line in f]
    except IOError:
      scores = []

    best_scores = sorted(scores, reverse=True)[:self.keep_best]

    if any(score_ < score for score_, _ in best_scores) or not best_scores:
      shutil.copy(os.path.join(self.checkpoint_dir, 'translate-{}'.format(step)),
                  os.path.join(self.checkpoint_dir, 'best-{}'.format(step)))

      if all(score_ < score for score_, _ in best_scores):
        path = os.path.abspath(os.path.join(self.checkpoint_dir, 'best'))
        try:  # remove old links
          os.remove(path)
        except OSError:
          pass
        # make symbolic links to best model
        os.symlink('{}-{}'.format(path, step), path)

      best_scores = sorted(best_scores + [(score, step)], reverse=True)

      for _, step_ in best_scores[self.keep_best:]:
        # remove checkpoints that are not in the top anymore
        try:
          os.remove(os.path.join(self.checkpoint_dir, 'best-{}'.format(step_)))
        except OSError:
          pass

    # save bleu scores
    scores.append((score, step))

    with open(score_filename, 'w') as f:
      for score_, step_ in scores:
        f.write('{} {}\n'.format(score_, step_))

  def initialize(self, sess, checkpoints=None, reset=False, reset_learning_rate=False):
    sess.run(tf.initialize_all_variables())
    if not reset:
      blacklist = ('learning_rate', 'dropout_keep_prob') if reset_learning_rate else ()
      load_checkpoint(sess, self.checkpoint_dir, blacklist=blacklist)

    if checkpoints is not None:  # load partial checkpoints
      for checkpoint in checkpoints:  # checkpoint files to load
        load_checkpoint(sess, None, checkpoint,
                        blacklist=('learning_rate', 'global_step', 'dropout_keep_prob'))

  def save(self, sess):
    save_checkpoint(sess, self.saver, self.checkpoint_dir, self.global_step)


def load_checkpoint(sess, checkpoint_dir, filename=None, blacklist=()):
  """ `checkpoint_dir` should be unique to this model
  if `filename` is None, we load last checkpoint, otherwise
    we ignore `checkpoint_dir` and load the given checkpoint file.
  """
  if filename is None:
    # load last checkpoint
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt is not None:
      filename = ckpt.model_checkpoint_path
  else:
    checkpoint_dir = os.path.dirname(filename)

  var_file = os.path.join(checkpoint_dir, 'vars.pkl')
  
  if os.path.exists(var_file):
    with open(var_file, 'rb') as f:
      var_names = cPickle.load(f)    
      variables = [var for var in tf.all_variables() if var.name in var_names]
  else:
    variables = tf.all_variables()
  
  # remove variables from blacklist
  variables = [var for var in variables if not any(var.name.startswith(prefix) for prefix in blacklist)]

  if filename is not None:
    utils.log('reading model parameters from {}'.format(filename))
    tf.train.Saver(variables).restore(sess, filename)

    utils.debug('retrieved parameters ({})'.format(len(variables)))
    for var in variables:
      utils.debug('  {} {}'.format(var.name, var.get_shape()))


def save_checkpoint(sess, saver, checkpoint_dir, step=None, name=None):
  """ `checkpoint_dir` should be unique to this model """
  var_file = os.path.join(checkpoint_dir, 'vars.pkl')
  name = name or 'translate'
  
  if not os.path.exists(checkpoint_dir):
    utils.log("creating directory {}".format(checkpoint_dir))
    os.makedirs(checkpoint_dir)  
  
  with open(var_file, 'wb') as f:
    var_names = [var.name for var in tf.all_variables()]
    cPickle.dump(var_names, f)
  
  utils.log('saving model to {}'.format(checkpoint_dir))
  checkpoint_path = os.path.join(checkpoint_dir, name)
  saver.save(sess, checkpoint_path, step, write_meta_graph=False)
  utils.log('finished saving model')
