from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import subprocess
import tempfile
import numpy as np
import math
import logging
import struct
import sys

from collections import namedtuple, OrderedDict
from contextlib import contextmanager
from itertools import izip

# special vocabulary symbols
_PAD = "_PAD"
_BOS = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _BOS, _EOS, _UNK]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


@contextmanager
def open_files(names, mode='r'):
  """ Safely open a list of files in a context manager.
  Example:
  >>> with open_files(['foo.txt', 'bar.csv']) as f:
  ...   pass
  """

  files = []
  try:
    for name_ in names:
      files.append(open(name_, mode=mode))
    yield files
  finally:
    for file_ in files:
      file_.close()


class AttrDict(dict):   # magical dict
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


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
  if os.path.exists(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path) as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.rstrip('\n') for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return namedtuple('vocab', 'vocab reverse')(vocab, rev_vocab)
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, character_level=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    character_level: consider sentence as a string of characters, and
      not as a string of words.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  sentence = sentence.strip() if character_level else sentence.split()
  return [vocabulary.get(w, UNK_ID) for w in sentence]


def get_filenames(data_dir, extensions, train_prefix, dev_prefix, lm_file=None, **kwargs):
  """ Last extension is always assumed to be the target """

  train_path = os.path.join(data_dir, train_prefix)
  dev_path = os.path.join(data_dir, dev_prefix)
  test_path = kwargs.get('decode')  # `decode` or `eval` or None
  test_path = test_path if test_path is not None else kwargs.get('eval')
  lm_path = lm_file

  train = ['{}.{}'.format(train_path, ext) for ext in extensions]
  dev = ['{}.{}'.format(dev_path, ext) for ext in extensions]
  test = test_path and ['{}.{}'.format(test_path, ext) for ext in extensions]
  vocab = [os.path.join(data_dir, 'vocab.{}'.format(ext)) for ext in extensions]

  filenames = namedtuple('filenames', ['train', 'dev', 'test', 'vocab', 'lm_path'])
  return filenames(**{k: v for k, v in vars().items() if k in filenames._fields})


def bleu_score(bleu_script, hypotheses, references):
  with tempfile.NamedTemporaryFile(delete=False) as f:
    for ref in references:
      f.write(ref + '\n')

  p = subprocess.Popen([bleu_script, f.name], stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))

  output, _ = p.communicate('\n'.join(hypotheses))

  m = re.match(r'BLEU = ([^,]*).*BP=([^,]*), ratio=([^,]*)', output)
  values = [float(m.group(i)) for i in range(1, 4)]

  return namedtuple('BLEU', ['score', 'penalty', 'ratio'])(*values)


def read_embeddings(filenames, extensions, vocab_sizes, embedding_size,
                    load_embeddings=None, norm_embeddings=None, **kwargs):
  embeddings = {}  

  for ext, vocab_size, vocab_path, filename, embedding_size_ in zip(extensions,
      vocab_sizes, filenames.vocab, filenames.embeddings, embedding_size):
    if load_embeddings is None or ext not in load_embeddings:
      continue

    with open(filename) as file_:
      lines = (line.split() for line in file_)
      _, size_ = next(lines)
      assert int(size_) == embedding_size_, 'wrong embedding size'
      embedding = np.zeros((vocab_size, size_), dtype="float32")

      d = dict((line[0], np.array(map(float, line[1:]))) for line in lines)

    vocab = initialize_vocabulary(vocab_path).vocab

    for word, index in vocab.iteritems():
      if word in d:
        embedding[index] = d[word]
      else:
        embedding[index] = np.random.uniform(-math.sqrt(3), math.sqrt(3), size_)

    if norm_embeddings:   # FIXME
      embedding /= np.linalg.norm(embedding)

    embeddings[ext] = embedding

  return embeddings


def read_binary_features(filename):
  """
  Reads a binary file containing vector features. First two numbers correspond to
  number of entries (lines), and dimension of the vectors.
  Each entry starts with a 32 bits integer indicating the number of frames, followed by
  (frames x dimension) 32 bits floating point numbers.

  @Returns: list of (frames x dimension) shaped arrays
  """
  all_feats = []

  with open(filename, 'rb') as f:
    lines, dim = struct.unpack('ii', f.read(8))
    for _ in xrange(lines):
      frames, = struct.unpack('i', f.read(4))
      n = frames * dim
      feats = struct.unpack('f' * n, f.read(4 * n))
      all_feats.append(list(np.array(feats).reshape(frames, dim)))

  return all_feats


def read_dataset(paths, extensions, vocabs, buckets, max_size=None, binary_input=None,
                 character_level=None):
  data_set = [[] for _ in buckets]

  line_reader = read_lines(paths, extensions, binary_input=binary_input)
  character_level = character_level or []

  for counter, inputs in enumerate(line_reader, 1):
    if max_size and counter >= max_size:
      break
    if counter % 100000 == 0:
      log("  reading data line {}".format(counter))

    inputs = [
      sentence_to_token_ids(input_, vocab.vocab, character_level=(ext in character_level))
      if vocab is not None and isinstance(input_, basestring)
      else input_
      for input_, vocab, ext in zip(inputs, vocabs, extensions)
    ]

    if not all(inputs):  # skip empty inputs
      continue

    for bucket_id, bucket in enumerate(buckets):
      if all(len(input_) < bucket_size for input_, bucket_size in zip(inputs, bucket)):
        data_set[bucket_id].append(inputs)
        break

  debug('files: {}'.format(' '.join(paths)))
  for bucket_id, data in enumerate(data_set):
    debug('  bucket {} size {}'.format(bucket_id, len(data)))

  return data_set


def read_lines(paths, extensions, binary_input=None):
  binary_input = binary_input or []

  iterators = [
    read_binary_features(filename) if ext in binary_input else open(filename)
    for ext, filename in zip(extensions, paths)
  ]

  return izip(*iterators)


def replace_unk(src_tokens, trg_tokens, trg_token_ids, lookup_dict):
  trg_tokens = list(trg_tokens)

  for trg_pos, trg_id in enumerate(trg_token_ids):
    if not 4 <= trg_id <= 19:  # UNK symbols range
      continue

    src_pos = trg_pos + trg_id - 11   # aligned source position (symbol 4 is UNK-7, symbol 19 is UNK+7)
    if 0 <= src_pos < len(src_tokens):
      src_word = src_tokens[src_pos]
      # look for a translation, otherwise take the source word itself (e.g. name or number)
      trg_tokens[trg_pos] = lookup_dict.get(src_word, src_word)
    else:   # aligned position is outside of source sentence, nothing we can do.
      trg_tokens[trg_pos] = _UNK

  return trg_tokens


def initialize_lookup_dict(lookup_dict_path):
  with open(lookup_dict_path) as f:
    return dict(line.split() for line in f)


def read_ngrams(lm_path, vocab):
  ngram_list = []
  with open(lm_path) as f:
    for line in f:
      line = line.strip()
      if re.match(r'\\\d-grams:', line):
        ngram_list.append({})
      elif not line or line == '\\end\\':
        continue
      elif ngram_list:
        arr = map(str.rstrip, line.split('\t'))
        ngram = arr.pop(1)
        ngram_list[-1][ngram] = list(map(float, arr))

  debug('loaded n-grams, order={}'.format(len(ngram_list)))

  ngrams = []
  mappings = {'<s>': _BOS, '</s>': _EOS, '<unk>': _UNK}

  for kgrams in ngram_list:
    d = {}
    for seq, probas in kgrams.iteritems():
      ids = tuple(vocab.get(mappings.get(w, w)) for w in seq.split())
      if any(id_ is None for id_ in ids):
        continue
      d[ids] = probas
    ngrams.append(d)
  return ngrams


def create_logger(log_file=None):                
  formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S')
  if log_file is not None:
    handler = logging.FileHandler(log_file)
  else:
    handler = logging.StreamHandler()
  handler.setFormatter(formatter)
  logger = logging.getLogger(__name__)
  logger.addHandler(handler)
  return logger


def log(msg, level=logging.INFO):
  logging.getLogger(__name__).log(level, msg)


def debug(msg): log(msg, level=logging.DEBUG)
def warn(msg): log(msg, level=logging.WARN)


def estimate_lm_probability(sequence, ngrams):
  """
  P(w_3 | w_1, w_2) =
      log_prob(w_1 w_2 w_3)             } if (w_1 w_2 w_3) in language model
      P(w_3 | w_2) + backoff(w_1 w_2)   } otherwise
  in case (w_1 w_2) has no backoff weight, a weight of 0.0 is used
  """
  sequence = tuple(sequence)
  order = len(sequence)
  assert 0 < order <= len(ngrams)
  ngrams_ = ngrams[order - 1]

  if sequence in ngrams_:
    return ngrams_[sequence][0]
  else:
    weights = ngrams[order - 2].get(sequence[:-1])
    backoff_weight = weights[1] if weights is not None and len(weights) > 1 else 0.0
    return estimate_lm_probability(sequence[1:], ngrams) + backoff_weight
