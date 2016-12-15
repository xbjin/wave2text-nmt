import os
import sys
import re
import subprocess
import tempfile
import logging
import random
import math

from collections import namedtuple
from contextlib import contextmanager

# special vocabulary symbols
_BOS = "<S>"
_EOS = "</S>"
_UNK = "<UNK>"

BOS_ID = 0
EOS_ID = 1
UNK_ID = 2


@contextmanager
def open_files(names, mode='r'):
    """ Safely open a list of files in a context manager.
    Example:
    >>> with open_files(['foo.txt', 'bar.csv']) as (f1, f2):
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


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    Example:
    >>> d = AttrDict(x=1, y=2)
    >>> d.x
    1
    >>> d.y = 3
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  # dark magic


def initialize_vocabulary(vocabulary_path):
    """
    Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {'dog': 0, 'cat': 1}, and a reversed vocabulary ['dog', 'cat'].

    :param vocabulary_path: path to the file containing the vocabulary.
    :return:
      the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path) as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.rstrip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return namedtuple('vocab', 'vocab reverse')(vocab, rev_vocab)
    else:
        raise ValueError("vocabulary file %s not found", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, character_level=False):
    """
    Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    :param sentence: a string, the sentence to convert to token-ids
    :param vocabulary: a dictionary mapping tokens to integers
    :param character_level: treat sentence as a string of characters, and
        not as a string of words
    :return: a list of integers, the token-ids for the sentence.
    """
    sentence = sentence.rstrip('\n') if character_level else sentence.split()
    return [vocabulary.get(w, UNK_ID) for w in sentence]


def get_filenames(data_dir, extensions, train_prefix, dev_prefix, vocab_prefix, **kwargs):
    """
    Get a bunch of file prefixes and extensions, and output the list of filenames to be used
    by the model.

    :param data_dir: directory where all the the data is stored
    :param extensions: list of file extensions, in the right order (last extension is always the target)
    :param train_prefix: name of the training corpus (usually 'train')
    :param dev_prefix: name of the dev corpus (usually 'dev')
    :param vocab_prefix: prefix of the vocab files (usually 'vocab')
    :param kwargs: optional contains an additional 'decode' or 'eval' parameter
    :return: namedtuple containing the filenames
    """
    train_path = os.path.join(data_dir, train_prefix)
    dev_path = os.path.join(data_dir, dev_prefix)
    vocab_path = os.path.join(data_dir, vocab_prefix)

    train = ['{}.{}'.format(train_path, ext) for ext in extensions]
    dev = ['{}.{}'.format(dev_path, ext) for ext in extensions]
    vocab = ['{}.{}'.format(vocab_path, ext) for ext in extensions]

    test = kwargs.get('decode')  # empty list means we decode from standard input
    if test is None:
        test = test or kwargs.get('eval')
        test = test or kwargs.get('align')

    filenames = namedtuple('filenames', ['train', 'dev', 'test', 'vocab'])
    return filenames(train, dev, test, vocab)


def bleu_score(hypotheses, references, script_dir):
    """
    Scoring function which calls the 'multi-bleu.perl' script.

    :param hypotheses: list of translation hypotheses
    :param references: list of translation references
    :param script_dir: directory containing the evaluation script
    :return: a pair (BLEU score, additional scoring information)
    """
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
        for ref in references:
            f.write(ref + '\n')

    bleu_script = os.path.join(script_dir, 'multi-bleu.perl')
    try:
        p = subprocess.Popen([bleu_script, f.name], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))
        output, _ = p.communicate(('\n'.join(hypotheses) + '\n').encode())
    finally:
        os.unlink(f.name)

    output = output.decode()

    m = re.match(r'BLEU = ([^,]*).*BP=([^,]*), ratio=([^,]*)', output)
    bleu, penalty, ratio = [float(m.group(i)) for i in range(1, 4)]

    return bleu, 'penalty={} ratio={}'.format(penalty, ratio)


def read_dataset(paths, vocabs, max_size=None, sort_by_length=False, max_seq_len=None):
    data_set = []

    line_reader = read_lines(paths)

    for counter, inputs in enumerate(line_reader, 1):
        if max_size and counter > max_size:
            break
        if counter % 100000 == 0:
            log("  reading data line {}".format(counter))

        inputs = [
            sentence_to_token_ids(input_, vocab.vocab)
            for input_, vocab in zip(inputs, vocabs)
        ]

        if not all(inputs):  # skip empty inputs
            continue
        # skip lines that are too long
        if max_seq_len and any(len(inputs_) > max_seq_len for inputs_ in inputs):
            continue

        data_set.append(inputs)

    debug('files: {}'.format(' '.join(paths)))
    debug('size: {}'.format(len(data_set)))

    if sort_by_length:
        data_set.sort(key=lambda lines: list(map(len, lines)))

    return data_set


def random_batch_iterator(data, batch_size):
    """
    The most basic form of batch iterator.

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :return: an iterator which yields random batches (indefinitely)
    """
    while True:
        yield random.sample(data, batch_size)


def cycling_batch_iterator(data, batch_size):
    """
    Indefinitely cycle through a dataset and yield batches (the dataset is shuffled
    at each new epoch)

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :return: an iterator which yields batches (indefinitely)
    """
    # random.shuffle(data)

    while True:
        # random.shuffle(data)
        batch_count = len(data) // batch_size
        for i in range(batch_count):
            yield data[i * batch_size:(i + 1) * batch_size]


def read_ahead_batch_iterator(data, batch_size, read_ahead=10, shuffle=True):
    """
    Same iterator as `cycling_batch_iterator`, except that it reads a number of batches
    at once, and sorts their content according to their size.

    This is useful for training, where all the sequences in one batch need to be padded
     to the same length as the longest sequence in the batch.

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :param read_ahead: number of batches to read ahead of time and sort (larger numbers
      mean faster training, but less random behavior)
    :return: an iterator which yields batches (indefinitely)
    """
    iterator = cycling_batch_iterator(data, batch_size)
    if read_ahead <= 1:
        while True:
            yield next(iterator)

    while True:
        batches = [next(iterator) for _ in range(read_ahead)]
        data_ = sorted(sum(batches, []), key=lambda lines: len(lines[-1]))
        batches = [data_[i * batch_size:(i + 1) * batch_size] for i in range(read_ahead)]
        if shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch


def get_batches(data, batch_size, batches=10, allow_smaller=True):
    """
    Segment `data` into a given number of fixed-size batches. The dataset is automatically shuffled.

    This function is for smaller datasets, when you need access to the entire dataset at once (e.g. dev set).
    For larger (training) datasets, where you may want to lazily iterate over batches
    and cycle several times through the entire dataset, prefer batch iterators
    (such as `cycling_batch_iterator`).

    :param data: the dataset to segment into batches (a list of data points)
    :param batch_size: the size of a batch
    :param batches: number of batches to return (0 for the largest possible number)
    :param allow_smaller: allow the last batch to be smaller
    :return: a list of batches (which are lists of `batch_size` data points)
    """
    if not allow_smaller:
        max_batches = len(data) // batch_size
    else:
        max_batches = int(math.ceil(len(data) / batch_size))

    if batches < 1 or batches > max_batches:
        batches = max_batches

    random.shuffle(data)
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(batches)]
    return batches


def read_lines(paths):
    iterators = [sys.stdin if filename is None else open(filename) for filename in paths]
    return zip(*iterators)


def create_logger(log_file=None):
    """
    Initialize global logger and return it.

    :param log_file: log to this file, or to standard output if None
    :return: created logger
    """
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
