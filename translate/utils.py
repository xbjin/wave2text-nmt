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
import sys

from collections import namedtuple
from contextlib import contextmanager

# special vocabulary symbols
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
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
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return namedtuple('vocab', 'vocab reverse')(vocab, rev_vocab)
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  return [vocabulary.get(w, UNK_ID) for w in sentence.split()]


def get_filenames(data_dir, src_ext, trg_ext, train_prefix, dev_prefix, embedding_prefix,
                  load_embeddings, multi_task=False, replace_unk=False, **kwargs):
  trg_ext = trg_ext[0]    # FIXME: for now
  
  train_path = os.path.join(data_dir, train_prefix)
  src_train = ["{}.{}".format(train_path, ext) for ext in src_ext]
  src_train_ids = ["{}.ids.{}".format(train_path, ext) for ext in src_ext]

  if multi_task:  # multi-task setting: one target file for each encoder
    trg_train = ["{}.{}.{}".format(train_path, ext, trg_ext) for ext in src_ext]
    trg_train_ids = ["{}.ids.{}.{}".format(train_path, ext, trg_ext) for ext in src_ext]
  else:
    trg_train = "{}.{}".format(train_path, trg_ext)
    trg_train_ids = "{}.ids.{}".format(train_path, trg_ext)

  dev_path = os.path.join(data_dir, dev_prefix)
  src_dev = ["{}.{}".format(dev_path, ext) for ext in src_ext]
  trg_dev = "{}.{}".format(dev_path, trg_ext)

  src_dev_ids = ["{}.ids.{}".format(dev_path, ext) for ext in src_ext]
  trg_dev_ids = "{}.ids.{}".format(dev_path, trg_ext)

  src_vocab = [os.path.join(data_dir, "vocab.{}".format(ext)) for ext in src_ext]
  trg_vocab = os.path.join(data_dir, "vocab.{}".format(trg_ext))

  test_path = kwargs.get('decode')  # `decode` or `eval` or None
  test_path = test_path if test_path is not None else kwargs.get('eval')

  src_test = ["{}.{}".format(test_path, ext) for ext in src_ext] if test_path is not None else None
  trg_test = "{}.{}".format(test_path, trg_ext) if test_path is not None else None
  lookup_dict = os.path.join(data_dir, 'lookup_dict') if replace_unk else None

  embedding_path = os.path.join(data_dir, embedding_prefix)
  embeddings = ['{}.{}'.format(embedding_path, ext) for ext in src_ext + [trg_ext]]

  filenames = namedtuple('filenames', ['src_train', 'trg_train', 'src_dev', 'trg_dev', 'src_vocab', 'trg_vocab',
                                       'src_train_ids', 'trg_train_ids', 'src_dev_ids', 'trg_dev_ids',
                                       'src_test', 'trg_test', 'lookup_dict', 'embeddings'])

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


def read_embeddings(filenames, src_ext, trg_ext, src_vocab_size, trg_vocab_size, size,
                    load_embeddings=None, fixed_embeddings=None, norm_embeddings=None, **kwargs):
  extensions = src_ext + trg_ext
  vocab_sizes = src_vocab_size + trg_vocab_size
  vocab_paths = filenames.src_vocab + [filenames.trg_vocab]

  embeddings = {}  

  for ext, vocab_size, vocab_path, filename in zip(extensions, vocab_sizes, vocab_paths, filenames.embeddings):
    embedding_type = namedtuple('embedding', 'value trainable')
    fixed = fixed_embeddings is not None and ext in fixed_embeddings

    if load_embeddings is None or ext not in load_embeddings:
      if fixed:
        embeddings[ext] = embedding_type(value=None, trainable=False)
      continue

    with open(filename) as file_:
      lines = (line.split() for line in file_)
      _, size_ = next(lines)
      assert int(size_) == size, 'wrong embedding size'
      embedding = np.zeros((vocab_size, size), dtype="float32")

      d = dict((line[0], np.array(map(float, line[1:]))) for line in lines)

    vocab = initialize_vocabulary(vocab_path).vocab

    for word, index in vocab.iteritems():
      if word in d:
        embedding[index] = d[word]
      else:
        embedding[index] = np.random.uniform(-math.sqrt(3), math.sqrt(3), size)
    
    embedding_type = namedtuple('embedding', 'value trainable')
    if norm_embeddings:
      embedding = embedding / np.linalg.norm(embedding)
      log('embedding of lang ' + ext + 'normalized')
    embeddings[ext] = embedding_type(embedding, not fixed)

  return embeddings


def read_dataset(source_paths, target_path, buckets, max_size=None):
  data_set = [[] for _ in buckets]

  filenames = source_paths + [target_path]
  with open_files(filenames) as files:

    for counter, lines in enumerate(zip(*files), 1):
      if max_size and counter >= max_size:
        break
      if counter % 100000 == 0:
        log("  reading data line {}".format(counter))

      ids = [map(int, line.split()) for line in lines]
      source_ids, target_ids = ids[:-1], ids[-1]
      
      target_ids.append(EOS_ID)

      if any(len(ids_) == 0 for ids_ in ids):  # skip empty lines
        continue

      for bucket_id, (source_size, target_size) in enumerate(buckets):
        if len(target_ids) < target_size and all(len(ids_) < source_size for ids_ in source_ids):
          data_set[bucket_id].append(source_ids + [target_ids])
          break

  debug('files: {}'.format(' '.join(filenames)))
  for bucket_id, data in enumerate(data_set):
    debug('  bucket {} size {}'.format(bucket_id, len(data)))

  return data_set


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
