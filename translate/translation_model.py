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
from translate import seq2seq_model, utils


class TranslationModel(object):
  def __init__(self, src_ext, trg_ext, parameters, embeddings, checkpoint_dir, learning_rate,
               learning_rate_decay_factor, multi_task=False, task_ratio=None, keep_best=1, lm_order=3):
    self.src_ext = src_ext
    self.trg_ext = trg_ext[0]
    self.buckets = [(5, 10), (10, 15), (20, 25), (51, 51)]
    self.checkpoint_dir = checkpoint_dir
    self.keep_best = keep_best
    self.lm_order = lm_order
    self.multi_task = multi_task
    self.parameters = parameters
    
    self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)    
    
    with tf.device('/cpu:0'):
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # main model
    self.model = seq2seq_model.Seq2SeqModel(src_ext, trg_ext, self.buckets, self.learning_rate, self.global_step,
                                            embeddings, **vars(parameters))
    self.models = []
    
    if multi_task:  # multi-task
      task_ratio = task_ratio or [1.0] * len(src_ext)
      self.task_ratio = [x / sum(task_ratio) for x in task_ratio]  # sum must be 1

      for ext, vocab_size in zip(src_ext, parameters.src_vocab_size):
        params = {k: v for k, v in vars(parameters).items() if k != 'src_vocab_size'}  # FIXME: ugly
        partial_model = seq2seq_model.Seq2SeqModel([ext], trg_ext, self.buckets, self.learning_rate, self.global_step,
                                                   embeddings, src_vocab_size=[vocab_size], reuse=True, **params)
        self.models.append(partial_model)
    else:  # multi-source
      self.task_ratio = [1.0]
      self.models.append(self.model)

    self.saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)

    self.src_vocabs = None
    self.trg_vocab = None

  def _read_data(self, filenames, max_train_size):
    utils.debug('reading training data')
    if self.multi_task:
      for model, src_train_ids, trg_train_ids in zip(self.models, filenames.src_train_ids, filenames.trg_train_ids):
        train_set = ([src_train_ids], trg_train_ids)
        model.read_data(train_set, self.buckets, max_train_size=max_train_size)
      
    else:
      train_set = (filenames.src_train_ids, filenames.trg_train_ids)
      self.model.read_data(train_set, self.buckets, max_train_size)
    
    utils.debug('reading development data')
    self.model.dev_set = utils.read_dataset(filenames.src_dev_ids, filenames.trg_dev_ids, self.buckets)

  def _read_vocab(self, filenames):
    self.src_vocabs = [utils.initialize_vocabulary(vocab_path) for vocab_path in filenames.src_vocab]
    self.trg_vocab = utils.initialize_vocabulary(filenames.trg_vocab)
    self.lookup_dict = filenames.lookup_dict and utils.initialize_lookup_dict(filenames.lookup_dict)
    self.ngrams = filenames.lm_path and utils.read_ngrams(filenames.lm_path, self.lm_order)

    # FIXME
    if any(len(vocab.reverse) != vocab_size for vocab, vocab_size in
           zip(self.src_vocabs + [self.trg_vocab], self.parameters.src_vocab_size + [self.parameters.trg_vocab_size])):
      utils.warn('warning: inconsistent vocabulary size')

  def initialize(self, sess, checkpoints=None, reset=False, reset_learning_rate=False):
    sess.run(tf.initialize_all_variables())
    if not reset:
      blacklist = ('learning_rate',) if reset_learning_rate else ()
      load_checkpoint(sess, self.checkpoint_dir, blacklist=blacklist)

    if checkpoints is not None:  # load partial checkpoints
      for checkpoint in checkpoints:  # checkpoint files to load
        load_checkpoint(sess, None, checkpoint,
                        blacklist=('learning_rate', 'global_step'))

  def train(self, sess, filenames, beam_size, steps_per_checkpoint, steps_per_eval=None, bleu_script=None,
            max_train_size=None, eval_output=None, remove_unk=False):
    utils.log('reading training and development data')
    self._read_data(filenames, max_train_size)
    
    # check read_data has been called
    previous_losses = []
      
    losses = [0.0] * len(self.models)
    times = [0.0] * len(self.models)
    steps = [0] * len(self.models)
    
    utils.log('starting training')
    while True:
      # pick random task according to task ratios
      i = np.random.choice(range(len(self.models)), p=self.task_ratio)
      model = self.models[i]

      start_time = time.time()
      step_loss = self._train_step(sess, model)

      times[i] += (time.time() - start_time)
      losses[i] += step_loss
      steps[i] += 1
      
      global_step = self.global_step.eval(sess)
      
      if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:        
        loss = sum(losses) / sum(steps)
        step_time = sum(times) / sum(steps)
        perplexity = math.exp(loss) if loss < 300 else float('inf')

        utils.log('global step {} learning rate {:.4f} step-time {:.2f} perplexity {:.2f}'.format(
          global_step, self.model.learning_rate.eval(), step_time, perplexity))
          
        # decay learning rate when loss is worse than last losses
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          utils.debug('decreasing learning rate')
          sess.run(self.learning_rate_decay_op)

        previous_losses.append(loss)
        
        if self.multi_task:  # detail per model
          perplexities = [math.exp(loss_ / steps_) if steps_ > 0 and loss_ / steps_ < 300 else float('inf')
                          for loss_, steps_ in zip(losses, steps)]
          step_times = [time_ / steps_ if steps_ > 0 else 0.0 for time_, steps_ in zip(times, steps)]                      
          detail = '\n'.join('  steps {} step-time {:.2f} perplexity {:.2f}'.format(steps_, step_time_, perplexity_)
              for steps_, step_time_, perplexity_ in zip(steps, step_times, perplexities))
          utils.log('details per model\n{}'.format(detail))
        
        losses = [0.0] * len(self.models)
        times = [0.0] * len(self.models)
        steps = [0] * len(self.models)      
        
        self._eval_step(sess, self.model)
        self.save(sess)
        
      if steps_per_eval and bleu_script and global_step % steps_per_eval == 0:
        utils.log('starting BLEU eval')
        output = '{}.{}'.format(eval_output, global_step)
        score = self.evaluate(sess, filenames, beam_size, bleu_script, on_dev=True, output=output,
                              remove_unk=remove_unk)
        self._manage_best_checkpoints(global_step, score)

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

    if any(score_ < score for score_, _ in best_scores):
      # TODO: check that best-* files are not deleted by saver
      shutil.copy(os.path.join(self.checkpoint_dir, 'translate-{}'.format(step)),
                  os.path.join(self.checkpoint_dir, 'best-{}'.format(step)))
      shutil.copy(os.path.join(self.checkpoint_dir, 'translate-{}.meta'.format(step)),
                  os.path.join(self.checkpoint_dir, 'best-{}.meta'.format(step)))

      if all(score_ < score for score_, _ in best_scores):
        path = os.path.abspath(os.path.join(self.checkpoint_dir, 'best'))
        try:  # remove old links
          os.remove(path)
          os.remove('{}.meta'.format(path))
        except OSError:
          pass
        # make symbolic links to best model
        os.symlink('{}-{}'.format(path, step), path)
        os.symlink('{}-{}.meta'.format(path, step), '{}.meta'.format(path))

      best_scores = sorted(best_scores + [(score, step)], reverse=True)

      for _, step_ in best_scores[self.keep_best:]:
        # remove checkpoints that are not in the top anymore
        try:
          os.remove(os.path.join(self.checkpoint_dir, 'best-{}'.format(step_)))
          os.remove(os.path.join(self.checkpoint_dir, 'best-{}.meta'.format(step_)))
        except OSError:
          pass

    # save bleu scores
    scores.append((score, step))

    with open(score_filename, 'w') as f:
      for score_, step_ in scores:
        f.write('{} {}\n'.format(score_, step_))

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

  def _decode_sentence(self, sess, src_sentences, beam_size=4, remove_unk=False):
    # See here: https://github.com/giancds/tsf_nmt/blob/master/tsf_nmt/nmt_models.py
    # or here: https://github.com/wchan/tensorflow/tree/master/speech4/models
    tokens = [sentence.split() for sentence in src_sentences]
    token_ids = [utils.sentence_to_token_ids(sentence, vocab.vocab)
                 for vocab, sentence in zip(self.src_vocabs, src_sentences)]
    max_len = self.buckets[-1][0] - 1

    if any(len(ids_) > max_len for ids_ in token_ids):
      len_ = max(map(len, token_ids))
      utils.warn("line is too long ({} tokens), truncating".format(len_))
      token_ids = [ids_[:max_len] for ids_ in token_ids]

    if beam_size <= 1 and not isinstance(sess, list):
      trg_token_ids = self.model.greedy_decoding(sess, token_ids)
    else:
      hypotheses, scores = self.model.beam_search_decoding(sess, token_ids, beam_size, ngrams=self.ngrams)
      trg_token_ids = hypotheses[0]   # first hypothesis is the highest scoring one

    # remove EOS symbols from output
    if utils.EOS_ID in trg_token_ids:
      trg_token_ids = trg_token_ids[:trg_token_ids.index(utils.EOS_ID)]

    trg_tokens = [self.trg_vocab.reverse[i] if i < len(self.trg_vocab.reverse) else utils._UNK
                  for i in trg_token_ids]

    if remove_unk:
      trg_tokens = [token for token in trg_tokens if token != utils._UNK]

    if self.lookup_dict is not None:
      trg_tokens = utils.replace_unk(tokens[0], trg_tokens, trg_token_ids, self.lookup_dict)

    return ' '.join(trg_tokens).replace('@@ ', '')  # merge subword units

  def decode(self, sess, filenames, beam_size, output=None, remove_unk=False):
    self._read_vocab(filenames)
    utils.debug('decoding, UNK replacement {}'.format('OFF' if self.lookup_dict is None else 'ON'))
      
    with utils.open_files(filenames.src_test) as files:
      output_file = None
      try:
        output_file = sys.stdout if output is None else open(output, 'w')
        
        for src_sentences in zip(*files):
          # trg_sentence = self._decode_sentence(sess, src_sentences)
          trg_sentence = self._decode_sentence(sess, src_sentences, beam_size, remove_unk)
          output_file.write(trg_sentence + '\n')
          output_file.flush()
          
      finally:
        if output_file is not None:
          output_file.close()

  def evaluate(self, sess, filenames, beam_size, bleu_script, on_dev=False, output=None, remove_unk=False):
    self._read_vocab(filenames)
    utils.debug('decoding, UNK replacement {}'.format('OFF' if self.lookup_dict is None else 'ON'))
    utils.debug('external language model {}'.format('OFF' if self.ngrams is None else 'ON'))

    src_filenames = filenames.src_dev if on_dev else filenames.src_test
    trg_filename = filenames.trg_dev if on_dev else filenames.trg_test

    with utils.open_files(src_filenames) as src_files, open(trg_filename) as trg_file:
      hypotheses = [self._decode_sentence(sess, src_sentences, beam_size, remove_unk)
                    for src_sentences in zip(*src_files)]
      references = [line.strip().replace('@@ ', '') for line in trg_file]
      
      bleu = utils.bleu_score(bleu_script, hypotheses, references)
      utils.log(bleu)
      if output is not None:
        with open(output, 'w') as f:
          f.writelines(line + '\n' for line in hypotheses)

      return bleu.score

  def save(self, sess):
    save_checkpoint(sess, self.saver, self.checkpoint_dir, self.global_step)

  def export_embeddings(self, sess, filenames, extensions, output_prefix):
    # FIXME
    utils.debug('exporting embeddings')
    vocab_filenames = dict(zip(self.src_ext + [self.trg_ext], filenames.src_vocab + [filenames.trg_vocab]))
    embeddings = self.model.get_embeddings(sess)

    for ext in extensions:
      vocab = utils.initialize_vocabulary(vocab_filenames[ext])
      embedding = embeddings[ext]
      output_filename = '{}.{}'.format(output_prefix, ext)

      with open(output_filename, 'w') as output_file:
        output_file.write('{} {}\n'.format(*embedding.shape))
        for i, vec in enumerate(embedding):
          word = vocab.reverse[i]
          vec_str = ' '.join(map(str, vec))
          output_file.write('{} {}\n'.format(word, vec_str))


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
  saver.save(sess, checkpoint_path, step)
  utils.log('finished saving model')
