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
import random
import numbers
import math
import wave

from collections import namedtuple
from contextlib import contextmanager
from itertools import izip
import matplotlib.pyplot as plt

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
  sentence = sentence.rstrip('\n') if character_level else sentence.split()
  return [vocabulary.get(w, UNK_ID) for w in sentence]


def get_filenames(data_dir, extensions, train_prefix, dev_prefix, vocab_prefix,
                  embedding_prefix, lm_file=None, **kwargs):
  """ Last extension is always assumed to be the target """
  train_path = os.path.join(data_dir, train_prefix)
  dev_path = [os.path.join(data_dir, prefix) for prefix in dev_prefix]
  vocab_path = os.path.join(data_dir, vocab_prefix)
  embedding_path = os.path.join(data_dir, embedding_prefix)
  test_path = kwargs.get('decode')  # `decode` or `eval` or None
  test_path = test_path if test_path is not None else kwargs.get('eval')
  test_path = test_path if test_path is not None else kwargs.get('align')
  lm_path = lm_file

  train = ['{}.{}'.format(train_path, ext) for ext in extensions]
  dev = [['{}.{}'.format(path, ext) for ext in extensions] for path in dev_path]
  vocab = ['{}.{}'.format(vocab_path, ext) for ext in extensions]
  embeddings = ['{}.{}'.format(embedding_path, ext) for ext in extensions]
  test = test_path and ['{}.{}'.format(test_path, ext) for ext in extensions]

  filenames = namedtuple('filenames', ['train', 'dev', 'test', 'vocab', 'lm_path', 'embeddings'])
  return filenames(train, dev, test, vocab, lm_path, embeddings)


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


def scoring(scoring_script, hypotheses, references):
  with tempfile.NamedTemporaryFile(delete=False) as f1, \
       tempfile.NamedTemporaryFile(delete=False) as f2:
    for ref in references:
      f1.write(ref + '\n')
    for hyp in hypotheses:
      f2.write(hyp + '\n')

  output = subprocess.check_output([scoring_script, f2.name, f1.name],
                                   stderr=open('/dev/null', 'w'))

  m = re.match(r'BLEU=(.*) NIST=(.*) TER=(.*) RATIO=(.*)', output)
  values = [float(m.group(i)) for i in range(1, 5)]

  return namedtuple('score', ['bleu', 'nist', 'ter', 'ratio'])(*values)


def read_embeddings(embedding_filenames, encoders_and_decoder, load_embeddings,
                    vocabs, norm_embeddings=False):
  for encoder_or_decoder, vocab, filename in zip(encoders_and_decoder,
                                                 vocabs,
                                                 embedding_filenames):
    name = encoder_or_decoder.name
    if not load_embeddings or name not in load_embeddings:
      encoder_or_decoder.embedding = None
      continue

    with open(filename) as file_:
      lines = (line.split() for line in file_)
      _, size_ = next(lines)
      size_ = int(size_)
      assert int(size_) == encoder_or_decoder.embedding_size, 'wrong embedding size'
      embedding = np.zeros((encoder_or_decoder.vocab_size, size_), dtype="float32")

      d = dict((line[0], np.array(map(float, line[1:]))) for line in lines)

    for word, index in vocab.vocab.iteritems():
      if word in d:
        embedding[index] = d[word]
      else:
        embedding[index] = np.random.uniform(-math.sqrt(3), math.sqrt(3), size_)

    if norm_embeddings:  # FIXME
      embedding /= np.linalg.norm(embedding)

    encoder_or_decoder.embedding = embedding


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


def read_dataset(paths, extensions, vocabs, max_size=None, binary_input=None,
                 character_level=None, sort_by_length=False):
  data_set = []

  line_reader = read_lines(paths, extensions, binary_input=binary_input)
  character_level = character_level or [False] * len(extensions)

  for counter, inputs in enumerate(line_reader, 1):
    if max_size and counter > max_size:
      break
    if counter % 100000 == 0:
      log("  reading data line {}".format(counter))

    inputs = [
      sentence_to_token_ids(input_, vocab.vocab, character_level=char_level)
      if vocab is not None and isinstance(input_, basestring)
      else input_
      for input_, vocab, ext, char_level in zip(inputs, vocabs, extensions, character_level)
    ]

    if not all(inputs):  # skip empty inputs
      continue

    data_set.append(inputs)   # TODO: filter too long

  debug('files: {}'.format(' '.join(paths)))
  debug('size: {}'.format(len(data_set)))

  if sort_by_length:
    data_set.sort(key=lambda lines: map(len, lines))

  return data_set


def bucket_batch_iterator(data, batch_size, bucket_count=10, key=None):
  if key is None:
    key = lambda x: x

  bucket_size = len(data) // bucket_count
  if bucket_size < batch_size:
    raise Exception('buckets are too small')

  data.sort(key=lambda lines: key(map(len, lines)))

  buckets = [
    data[i * bucket_size:(i + 1) * bucket_size] for i in range(bucket_count)
  ]
  buckets[-1] = data[(bucket_count - 1) * bucket_size:]

  while True:
    bucket = random.choice(buckets)
    yield random.sample(bucket, batch_size)


def random_batch_iterator(data, batch_size):
  while True:
    yield random.sample(data, batch_size)


def cycling_batch_iterator(data, batch_size):
  while True:
    random.shuffle(data)

    batch_count = len(data) // batch_size
    for i in range(batch_count):
      yield data[i * batch_size:(i + 1) * batch_size]


def sequential_sorted_batch_iterator(data, batch_size, read_ahead=10):
  iterator = cycling_batch_iterator(data, batch_size)
  while True:
    batches = [next(iterator) for _ in range(read_ahead)]
    data_ = sorted(sum(batches, []), key=lambda lines: len(lines[-1]))
    batches = [data_[i * batch_size:(i + 1) * batch_size] for i in range(read_ahead)]
    for batch in batches:
      yield batch


def random_sorted_batch_iterator(data, batch_size):
  # this iterator is seriously bad (prefer the read_ahead iterator)
  data.sort(key=lambda lines: len(lines[-1]))  # sort according to output length
  while True:
    i = random.randrange(len(data) - batch_size)
    batch = data[i:i + batch_size]
    yield batch


def get_batches(data, batch_size, batches=10, allow_smaller=True):
  if not allow_smaller:
    max_batches = len(data) // batch_size
  else:
    max_batches = int(math.ceil(len(data) / batch_size))

  if batches < 1 or batches > max_batches:
    batches = max_batches

  random.shuffle(data)
  batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(batches)]
  return batches


def bucket_iterator(data, batch_size, buckets):
  data_ = [[] for _ in buckets]
  for lines in data:
    try:
      i = next(i for i, bucket in enumerate(buckets)
               if all(len(line) <= b for line, b in zip(lines, bucket)))
    except StopIteration:
      continue

    data_[i].append(lines)

  p = [len(bucket) / sum(map(len, data_)) for bucket in data_]
  p_ = [sum(p[:i + 1]) for i in range(len(p))]
  while True:
    r = random.random()
    bucket_id = next(i for i, r_ in enumerate(p_) if r_ >= r)
    sample = [random.choice(data_[bucket_id]) for _ in range(batch_size)]
    yield sample


def read_lines(paths, extensions, binary_input=None):
  binary_input = binary_input or [False] * len(extensions)

  iterators = [
    read_binary_features(filename) if binary else open(filename)
    for ext, filename, binary in zip(extensions, paths, binary_input)
  ]

  return izip(*iterators)


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
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
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


def estimate_lm_score(sequence, ngrams):
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
    return estimate_lm_score(sequence[1:], ngrams) + backoff_weight


def advanced_shape(list_or_array):
  """
  Utility function to quickly get the shape of a list of arrays
  """
  if isinstance(list_or_array, numbers.Number):
    return str(type(list_or_array))
  if isinstance(list_or_array, np.ndarray):
    return 'array({}, {})'.format(', '.join(map(str, list_or_array.shape)), list_or_array.dtype)
  elif isinstance(list_or_array, list):
    return 'list({}, {})'.format(len(list_or_array), advanced_shape(list_or_array[0]))
  elif isinstance(list_or_array, tuple):
    return 'tuple({}, {})'.format(len(list_or_array), advanced_shape(list_or_array[0]))
  else:
    raise Exception('error: unknown type: {}'.format(type(list_or_array)))


def heatmap(xlabels=None, ylabels=None, weights=None, output_file=None):
  xlabels = xlabels or []
  ylabels = ylabels or []

  xlabels = [label.decode('utf-8') for label in xlabels]
  ylabels = [label.decode('utf-8') for label in ylabels]

  fig, ax = plt.subplots()
  heatmap_ = ax.pcolor(weights, cmap=plt.cm.Greys)
  ax.set_frame_on(False)

  plt.colorbar(mappable=heatmap_)

  # put the major ticks at the middle of each cell
  ax.set_yticks(np.arange(weights.shape[0]) + 0.5, minor=False)
  ax.set_xticks(np.arange(weights.shape[1]) + 0.5, minor=False)
  ax.invert_yaxis()
  ax.xaxis.tick_top()

  ax.set_xticklabels(xlabels, minor=False)
  ax.set_yticklabels(ylabels, minor=False)
  # plt.xticks(rotation=45, fontsize=12, ha='left')
  plt.xticks(rotation=90, fontsize=14)
  plt.yticks(fontsize=14)
  plt.tight_layout()
  ax.set_aspect('equal')

  ax.grid(False)
  ax = plt.gca()  # turn off all the ticks
  return fig


def plot_waveform(filename):
  from pylab import fromstring
  with wave.open(filename) as f:
    sound_info = f.readframes(-1)
    sound_info = fromstring(sound_info, 'int16')
    plt.plot(sound_info)
