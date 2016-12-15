import tensorflow as tf
import os
import pickle
import time
import sys
import math
import numpy as np
import shutil
from translate import utils
from translate.seq2seq_model import Seq2SeqModel


class BaseTranslationModel(object):
    def __init__(self, name, checkpoint_dir, keep_best=1):
        self.name = name
        self.keep_best = keep_best
        self.checkpoint_dir = checkpoint_dir
        self.saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)

    def manage_best_checkpoints(self, step, score):
        score_filename = os.path.join(self.checkpoint_dir, 'scores.txt')
        # try loading previous scores
        try:
            with open(score_filename) as f:
                # list of pairs (score, step)
                scores = [(float(line.split()[0]), int(line.split()[1])) for line in f]
        except IOError:
            scores = []

        if any(step_ >= step for _, step_ in scores):
            utils.warn('inconsistent scores.txt file')

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
                # copy of best model
                shutil.copy('{}-{}'.format(path, step), path)

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

    def initialize(self, sess, checkpoints=None, reset=False, init_from_blocks=None):
        sess.run(tf.initialize_all_variables())
        if checkpoints:  # load partial checkpoints
            for checkpoint in checkpoints:  # checkpoint files to load
                load_checkpoint(sess, None, checkpoint,
                                blacklist=('learning_rate', 'global_step', 'dropout_keep_prob'))
        elif not reset:
            load_checkpoint(sess, self.checkpoint_dir, blacklist=())

        if init_from_blocks is not None:
            utils.log('initializing variables from block')
            with open(init_from_blocks, 'rb') as f:
                block_vars = pickle.load(f, encoding='latin1')

            variables = {var_.name[:-2]: var_ for var_ in tf.all_variables()}

            for var_names, axis, value in block_vars:
                if 'decoder_en/attention_fr/v_a' in var_names:
                    value = np.squeeze(value)

                sections = [variables[name].get_shape()[axis].value for name in var_names]
                values = np.split(value, sections[:-1], axis=axis)

                for var_name, value in zip(var_names, values):
                    utils.debug(var_name)

                    var_ = variables[var_name]
                    print(var_.get_shape(), value.shape)
                    assert tuple(x.value for x in var_.get_shape()) == value.shape, \
                           'wrong shape for var: {}'.format(var_name)
                    sess.run(var_.assign(value))

            utils.log('read {} variables'.format(sum(len(var_names) for var_names, _, _ in block_vars)))


    def save(self, sess):
        save_checkpoint(sess, self.saver, self.checkpoint_dir, self.global_step)


class TranslationModel(BaseTranslationModel):
    def __init__(self, name, encoder, decoder, checkpoint_dir, learning_rate,
                 batch_size, keep_best=1,
                 optimizer='sgd', max_input_len=None, **kwargs):
        self.batch_size = batch_size
        src_ext = encoder.get('ext') or encoder.name
        trg_ext = decoder.get('ext') or decoder.name
        self.extensions = [src_ext, trg_ext]
        self.max_input_len = max_input_len
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate', dtype=tf.float32)

        with tf.device('/cpu:0'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.filenames = utils.get_filenames(extensions=self.extensions, **kwargs)

        utils.debug('reading vocabularies')
        self._read_vocab()
        if encoder.vocab_size <= 0:
            encoder.vocab_size = len(self.vocabs[0].reverse)
        if decoder.vocab_size <= 0:
            decoder.vocab_size = len(self.vocabs[1].reverse)

        # main model
        utils.debug('creating model {}'.format(name))
        self.model = Seq2SeqModel(encoder, decoder, self.learning_rate, self.global_step, optimizer=optimizer,
                                  max_input_len=max_input_len, **kwargs)

        super(TranslationModel, self).__init__(name, checkpoint_dir, keep_best)

        self.batch_iterator = None
        self.dev_batches = None

    def read_data(self, max_train_size, max_dev_size, read_ahead=10):
        utils.debug('reading training data')
        train_set = utils.read_dataset(self.filenames.train, self.vocabs, max_size=max_train_size,
                                       max_seq_len=self.max_input_len)
        self.batch_iterator = utils.read_ahead_batch_iterator(train_set, self.batch_size, read_ahead=read_ahead,
                                                              shuffle=False)

        utils.debug('reading development data')
        dev_set = utils.read_dataset(self.filenames.dev, self.vocabs, max_size=max_dev_size)
        # subset of the dev set whose perplexity is periodically evaluated
        self.dev_batches = utils.get_batches(dev_set, batch_size=self.batch_size, batches=-1)

    def _read_vocab(self):
        # don't try reading vocabulary for encoders that take pre-computed features
        self.vocabs = [
            utils.initialize_vocabulary(vocab_path)
            for ext, vocab_path in zip(self.extensions, self.filenames.vocab)
        ]
        self.src_vocab, self.trg_vocab = self.vocabs

    def train(self, sess, beam_size, steps_per_checkpoint, steps_per_eval=None, max_train_size=None,
              max_dev_size=None, eval_output=None, max_steps=0, script_dir='scripts', read_ahead=10, eval_burn_in=0,
              **kwargs):
        utils.log('reading training and development data')

        self.read_data(max_train_size, max_dev_size, read_ahead=read_ahead)
        loss, time_, steps = 0, 0, 0

        global_step = self.global_step.eval(sess)
        for _ in range(global_step):   # read all the data up to this step
            next(self.batch_iterator)

        utils.log('starting training')
        while True:
            start_time = time.time()
            loss += self.train_step(sess)
            time_ += (time.time() - start_time)
            steps += 1
            global_step = self.global_step.eval(sess)

            if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:

                loss_ = loss / steps
                step_time_ = time_ / steps

                utils.log('{} step {} step-time {:.2f} loss {:.2f}'.format(
                    self.name, global_step, step_time_, loss_))

                loss, time_, steps = 0, 0, 0
                self.eval_step(sess)
                self.save(sess)

            if steps_per_eval and global_step % steps_per_eval == 0 and 0 <= eval_burn_in <= global_step:
                if eval_output is None:
                    output = None
                else:
                    output = '{}.{}.{}'.format(eval_output, self.name, self.global_step.eval(sess))

                score = self.evaluate(
                    sess, beam_size, on_dev=True, output=output, script_dir=script_dir, max_dev_size=max_dev_size)
                self.manage_best_checkpoints(global_step, score)

            if 0 < max_steps <= global_step:
                utils.log('finished training')
                return

    def train_step(self, sess):
        return self.model.step(sess, next(self.batch_iterator))

    def eval_step(self, sess):
        # compute perplexity on dev set
        eval_loss = sum(
            self.model.step(sess, batch, forward_only=True) * len(batch)
            for batch in self.dev_batches
        )
        eval_loss /= sum(map(len, self.dev_batches))

        utils.log("  eval: loss {:.2f}".format(eval_loss))

    def _decode_sentence(self, sess, src_sentence, beam_size=1, remove_unk=False):
        token_ids = utils.sentence_to_token_ids(src_sentence, self.src_vocab.vocab)

        if beam_size <= 1 and not isinstance(sess, list):
            trg_token_ids = self.model.greedy_decoding(sess, token_ids)
        else:
            hypotheses, scores = self.model.beam_search_decoding(sess, token_ids, beam_size)
            trg_token_ids = hypotheses[0]  # first hypothesis is the highest scoring one

        # remove EOS symbols from output
        if utils.EOS_ID in trg_token_ids:
            trg_token_ids = trg_token_ids[:trg_token_ids.index(utils.EOS_ID)]

        trg_tokens = [self.trg_vocab.reverse[i] if i < len(self.trg_vocab.reverse) else utils._UNK
                      for i in trg_token_ids]

        if remove_unk:
            trg_tokens = [token for token in trg_tokens if token != utils._UNK]

        return ' '.join(trg_tokens).replace('@@ ', '')  # merge subword units

    def decode(self, sess, beam_size, output=None, remove_unk=False, **kwargs):
        utils.log('starting decoding')

        output_file = None
        try:
            output_file = sys.stdout if output is None else open(output, 'w')

            for lines in utils.read_lines(self.filenames.test):
                trg_sentence = self._decode_sentence(sess, lines, beam_size, remove_unk)
                output_file.write(trg_sentence + '\n')
                output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()

    def evaluate(self, sess, beam_size, on_dev=True, output=None, remove_unk=False, max_dev_size=None,
                 script_dir='scripts', **kwargs):
        """
        :param on_dev: if True, evaluate the dev corpus, otherwise evaluate the test corpus
        :param output: save the hypotheses to this file
        :param remove_unk: remove the UNK symbols from the output
        :param max_dev_size: maximum number of lines to read from dev files
        :param script_dir: parameter of scoring functions
        :return: score
        """
        utils.log('starting decoding')
        assert on_dev or len(self.filenames.test) == len(self.extensions)

        filenames = self.filenames.dev if on_dev else self.filenames.test

        lines = list(utils.read_lines(filenames))
        if on_dev and max_dev_size:
            lines = lines[:max_dev_size]

        hypotheses = []
        references = []

        output_file = None
        try:
            output_file = open(output, 'w') if output is not None else None

            for src_sentence, trg_sentence in lines:
                hypotheses.append(self._decode_sentence(sess, src_sentence, beam_size, remove_unk))
                references.append(trg_sentence.strip().replace('@@ ', ''))
                if output_file is not None:
                    output_file.write(hypotheses[-1] + '\n')
                    output_file.flush()

        finally:
            if output_file is not None:
                output_file.close()

        # main scoring function (used to choose which checkpoints to keep)
        # default is utils.bleu_score
        score, score_summary = utils.bleu_score(hypotheses, references, script_dir)

        # print the scoring information
        score_info = []
        if self.name is not None:
            score_info.append(self.name)
        score_info.append('score={}'.format(score))
        if score_summary:
            score_info.append(score_summary)

        utils.log(' '.join(map(str, score_info)))
        return score


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
            var_names = pickle.load(f)
            variables = [var for var in tf.all_variables() if var.name in var_names]
    else:
        variables = tf.all_variables()

    # remove variables from blacklist
    variables = [var for var in variables if not any(prefix in var.name for prefix in blacklist)]

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
        pickle.dump(var_names, f)

    utils.log('saving model to {}'.format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, name)
    saver.save(sess, checkpoint_path, step, write_meta_graph=False)
    utils.log('finished saving model')
