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

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import subprocess
import shlex
import tempfile
import numpy as np
import tensorflow as tf
import math
import logging

import sys


from collections import namedtuple
from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    logging.info("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    logging.info("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    logging.info("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "w") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "giga-fren.release2")
  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _WMT_ENFR_TRAIN_URL)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
    gunzip_file(train_path + ".en.gz", train_path + ".en")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
    print("Extracting tgz file %s" % dev_file)
    with tarfile.open(dev_file, "r:gz") as dev_tar:
      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
      en_dev_file.name = dev_name + ".en"
      dev_tar.extract(fr_dev_file, directory)
      dev_tar.extract(en_dev_file, directory)
  return dev_path


def no_tokenizer(sentence):
  return sentence.split()


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    logging.info("Creating vocabulary %s from data %s" %
                 (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          logging.info("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    logging.info("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            logging.info("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path = get_wmt_enfr_train_set(data_dir)
  dev_path = get_wmt_enfr_dev_set(data_dir)

  # Create vocabularies of the appropriate sizes. (vocab40000)
  fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocabulary_size)
  en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
  create_vocabulary(fr_vocab_path, train_path + ".fr", fr_vocabulary_size)
  create_vocabulary(en_vocab_path, train_path + ".en", en_vocabulary_size)

  # Create token ids for the training data. (giga-fren.release2.ids40000)
  fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path)
  data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path)

  # Create token ids for the development data. (newstest2013.ids40000)
  fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path)
  data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path)

  return (en_train_ids_path, fr_train_ids_path,
          en_dev_ids_path, fr_dev_ids_path,
          en_vocab_path, fr_vocab_path)


def extract_filenames(FLAGS):
  """ Add filenames to the FLAGS namespace """
  src_ext = FLAGS.src_ext.split(',')
  trg_ext = FLAGS.trg_ext

  FLAGS.train_path = os.path.join(FLAGS.data_dir, FLAGS.train_corpus)

  FLAGS.src_train = ["{}.{}".format(FLAGS.train_path, ext) for ext in src_ext]
  FLAGS.trg_train = "{}.{}".format(FLAGS.train_path, trg_ext)

  FLAGS.src_train_ids = [
    "{}.ids{}.{}".format(FLAGS.train_path, FLAGS.src_vocab_size, ext)
    for ext in src_ext]
  FLAGS.trg_train_ids = "{}.ids{}.{}".format(
    FLAGS.train_path, FLAGS.trg_vocab_size, trg_ext)

  FLAGS.dev_path = os.path.join(FLAGS.data_dir, FLAGS.dev_corpus)
  FLAGS.src_dev = ["{}.{}".format(FLAGS.dev_path, ext) for ext in src_ext]
  FLAGS.trg_dev = "{}.{}".format(FLAGS.dev_path, trg_ext)

  FLAGS.src_dev_ids = [
    "{}.ids{}.{}".format(FLAGS.dev_path, FLAGS.src_vocab_size, ext)
    for ext in src_ext]
  FLAGS.trg_dev_ids = "{}.ids{}.{}".format(
    FLAGS.dev_path, FLAGS.trg_vocab_size, trg_ext)

  FLAGS.src_vocab = [os.path.join(FLAGS.data_dir, "vocab{}.{}".format(
    FLAGS.src_vocab_size, ext)) for ext in src_ext]

  FLAGS.trg_vocab = os.path.join(FLAGS.data_dir, "vocab{}.{}".format(
    FLAGS.trg_vocab_size, trg_ext))


def prepare_data(FLAGS):
  """ Create vocabularies and tokenize data. """

  tokenizer = no_tokenizer if not FLAGS.tokenize else None

  #if FLAGS.download:  # FIXME
  #  get_wmt_enfr_train_set(FLAGS.data_dir)
  #  get_wmt_enfr_dev_set(FLAGS.data_dir)

  for vocab, train, train_ids, dev, dev_ids in zip(FLAGS.src_vocab,
      FLAGS.src_train, FLAGS.src_train_ids, FLAGS.src_dev, FLAGS.src_dev_ids):
    create_vocabulary(vocab, train, FLAGS.src_vocab_size, tokenizer=tokenizer)
    data_to_token_ids(train, train_ids, vocab, tokenizer=tokenizer)
    data_to_token_ids(dev, dev_ids, vocab, tokenizer=tokenizer)


  create_vocabulary(FLAGS.trg_vocab, FLAGS.trg_train, FLAGS.trg_vocab_size,
                    tokenizer=tokenizer)
  data_to_token_ids(FLAGS.trg_train, FLAGS.trg_train_ids, FLAGS.trg_vocab,
                    tokenizer=tokenizer)
  data_to_token_ids(FLAGS.trg_dev, FLAGS.trg_dev_ids, FLAGS.trg_vocab,
                    tokenizer=tokenizer)


def bleu_score(bleu_script, hypotheses, references):
  with tempfile.NamedTemporaryFile(delete=False) as f:
    for ref in references:
      f.write(ref + '\n')

  p = subprocess.Popen([bleu_script, f.name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                       stderr=open('/dev/null', 'w'))

  output, _ = p.communicate('\n'.join(hypotheses))

  m = re.match(r'BLEU = ([^,]*).*BP=([^,]*), ratio=([^,]*)', output)
  values = [float(m.group(i)) for i in range(1, 4)]

  return namedtuple('BLEU', ['score', 'penalty', 'ratio'])(*values)


def extract_embedding(FLAGS):
    
  src_ext = FLAGS.src_ext.split(',')
  trg_ext = FLAGS.trg_ext  
  exts = src_ext + [trg_ext]
    
  if FLAGS.embedding_train:
      embedding_train = FLAGS.embedding_train.split(',')
  else:   
      embedding_train = [True for _ in exts] 
    
  vocabs = FLAGS.src_vocab + [FLAGS.trg_vocab]
  
  FLAGS.embeddings = [None for _ in exts]

  if not FLAGS.embedding:
    return

  for i, (ext, vocab_path) in enumerate(zip(exts,
                vocabs)):
    filename = os.path.join(FLAGS.data_dir, "{}.{}".format(FLAGS.embedding,
                                                           ext))
    # if embedding is not given for this language, skip
    if not os.path.isfile(filename):
      continue

    with open(filename) as file_:
      lines = (line.split() for line in file_)
      _, size = next(lines)
      size = int(size)

      embeddings = np.zeros((FLAGS.src_vocab_size, size), dtype="float32")
      d = dict((line[0], np.array(map(float, line[1:]))) for line in lines)

    vocab, _ = initialize_vocabulary(vocab_path)

    for word, index in vocab.iteritems():
      if word in d:
        embeddings[index] = d[word]
      else:
        embeddings[index] = np.random.uniform(-math.sqrt(3), math.sqrt(3), size)

    # sets embedding matrix as initial value
    FLAGS.embeddings[i] = tf.Variable(embeddings,
                                      name="custom_embedding_" + ext, trainable=(embedding_train[i]=='True'))
    
      
  #FLAGS.embeddings[-1]) is decoder embedding
  #print(FLAGS.embeddings)
