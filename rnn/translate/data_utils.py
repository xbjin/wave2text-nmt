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

import os
import re
import tempfile
import subprocess
from collections import namedtuple

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
_DIGIT_RE = re.compile(r"\d")


def preprocess(sentence, normalize_digits=True):
  """ Apply whatever preprocessing there is to do (tokenizing, mapping digits, etc.) """
  if normalize_digits:
    return ' '.join(re.sub(_DIGIT_RE, "0", w) for w in sentence.split())
  else:
    return sentence


""" split les phrases en mot et pour chaque mot trouve incremente son index de 1"""
""" retourne la liste des max_vocabulary_size mots les plus utilises"""
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, normalize_digits=True):
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
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      for counter, line in enumerate(f, 1):
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = line.split()
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w   # FIXME: preprocessing
          vocab[word] = vocab.get(word, 0) + 1
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
    vocab = dict((x, y) for (y, x) in enumerate(rev_vocab))
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  words = sentence.split()
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        for counter, line in enumerate(data_file, 1):
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, normalize_digits)
          tokens_file.write(" ".join(str(tok) for tok in token_ids) + "\n")


def prepare_data(flags):
  create_vocabulary(flags.src_vocab, flags.src_train, flags.src_vocab_size)
  create_vocabulary(flags.trg_vocab, flags.trg_train, flags.trg_vocab_size)

  data_to_token_ids(flags.src_train, flags.src_train_ids, flags.src_vocab)
  data_to_token_ids(flags.trg_train, flags.trg_train_ids, flags.trg_vocab)

  data_to_token_ids(flags.src_dev, flags.src_dev_ids, flags.src_vocab)
  data_to_token_ids(flags.trg_dev, flags.trg_dev_ids, flags.trg_vocab)


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
