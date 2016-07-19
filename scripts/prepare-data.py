#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import izip, islice
from random import shuffle
from contextlib import contextmanager
from collections import Counter
from functools import partial
import itertools
import argparse
import subprocess
import tempfile
import os
import logging
import sys
import shutil
import codecs
import shlex


help_msg = """\
Prepare a parallel corpus for Neural Machine Translation.

If a single corpus is specified, it will be split into train/dev/test corpora
according to the given train/dev/test sizes.

Additional pre-processing can be applied to these files, using external (Moses)
scripts, such as tokenization, punctuation normalization or lowercasing.
The corpus can be shuffled, and too long or too short sentences removed.

Usage example:
    scripts/prepare-data.py data/news fr en output --dev-corpus data/news-dev\
    --test-size 6000 --max 0 --lowercase --shuffle

This example will create 6 files in `output/`: train.fr, train.en, test.fr,\
 test.en, dev.fr and dev.en. These files will be tokenized and lowercased and\
 empty lines will be filtered out. `test` files will contain 6000 lines\
 from input corpus `data/news`, and `train` will contain the remaining\
 lines. `dev` files will contain the (processed) lines read from\
 `data/news-dev`. These three output corpora will be shuffled.
"""

_PAD = '_PAD'
_GO = '_GO'
_EOS = '_EOS'
_UNK = '_UNK'

_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

temporary_files = []


open = partial(codecs.open, encoding='utf8')


@contextmanager
def open_files(names, mode='r'):
    files = []
    try:
        for name_ in names:
            files.append(codecs.open(name_, mode=mode))
        yield files
    finally:
        for file_ in files:
            file_.close()


@contextmanager
def open_temp_files(num=1, mode='w', delete=False):
    files = []
    try:
        for _ in range(num):
            files.append(tempfile.NamedTemporaryFile(mode=mode, delete=delete))
            if not delete:
                temporary_files.append(files[-1].name)
        yield files
    finally:
        for file_ in files:
            file_.close()


def read_vocabulary(filename):
    with open(filename) as vocab_file:
        words = [line.strip() for line in vocab_file]
        return dict(map(reversed, enumerate(words)))


def create_vocabulary(filename, output_filename, size, character_level=False):
    logging.info('creating vocabulary {} from {}'.format(output_filename,
                                                         filename))
    vocab = Counter()
    with open(filename) as input_file, \
         open(output_filename, 'w') as output_file:
        for line in input_file:
            line = line.strip() if character_level else line.split()

            for w in line:
                vocab[w] += 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if 0 < size < len(vocab_list):
            vocab_list = vocab_list[:size]

        output_file.writelines(w + '\n' for w in vocab_list)

    return dict(map(reversed, enumerate(vocab_list)))


def process_file(filename, lang, args):
    logging.info('processing ' + filename)

    with open_temp_files(num=1) as output_, open(filename) as input_:
        output_, = output_

        def path_to(script_name):
            if args.scripts is None:
                return script_name   # assume script is in PATH
            else:
                return os.path.join(args.scripts, script_name)

        processes = [['cat']]   # just copy file if there is no other operation

        if args.normalize_punk:
            processes.append([path_to('normalize-punctuation.perl'), '-l',
                              lang])
            # replace html entities: FIXME
            # processes.append(shlex.split("perl -MHTML::Entities -pe 'decode_entities($_);'"))
        if args.normalize_moses:
            processes.append(['sed', 's/|//g'])
        if args.subwords:
            processes.append(['sed', 's/@\\+/@/g'])  # @@ is used as delimiter for subwords

        if args.tokenize:
            processes.append([path_to('tokenizer.perl'), '-l', lang, '-threads',
                              str(args.threads)])
        if args.lowercase:
            processes.append([path_to('lowercase.perl')])
        if args.normalize_digits:
            processes.append(['sed', 's/[[:digit:]]/0/g'])
        if args.escape_special_chars:
            processes.append([path_to('escape-special-chars.perl')])

        ps = None

        for i, process in enumerate(processes):
            stdout = output_ if i == len(processes) - 1 else subprocess.PIPE
            stdin = input_ if i == 0 else ps.stdout

            ps = subprocess.Popen(process, stdin=stdin, stdout=stdout,
                                  stderr=open('/dev/null', 'w'))

        ps.wait()
        return output_.name


def filter_corpus(filenames, args):
    with open_files(filenames) as input_files, \
         open_temp_files(len(filenames)) as output_files:
        for lines in izip(*input_files):
            if all(min_ <= len(line.split()) <= max_ for line, min_, max_
                   in zip(lines, args.min, args.max)):
                for line, output_file in zip(lines, output_files):
                    output_file.write(line)

        return [f.name for f in output_files]


def process_corpus(filenames, args):
    filenames = [process_file(filename, lang, args)
                 for lang, filename in zip(args.lang, filenames)]

    with open_files(filenames) as input_files, \
         open_temp_files(len(filenames)) as output_files:

        # (lazy) sequence of sentence tuples
        all_lines = (lines for lines in izip(*input_files) if
                     all(min_ <= len(line.split()) <= max_ for line, min_, max_
                         in zip(lines, args.min, args.max)))

        if args.remove_duplicate_lines:
            seen_lines = [set() for _ in filenames]
            lines = []
            for line_tuple in all_lines:
                if not any(line in seen_lines_ for line, seen_lines_ in
                           zip(line_tuple, seen_lines)):
                    lines.append(line_tuple)
            all_lines = lines
        elif args.remove_duplicates:
            all_lines = list(set(all_lines))

        if args.shuffle:
            all_lines = list(all_lines)  # not lazy anymore
            shuffle(all_lines)

        for lines in all_lines:  # keeps it lazy if no shuffle
            for line, output_file in zip(lines, output_files):
                output_file.write(line)

        return [f.name for f in output_files]


def split_corpus(filenames, sizes):
    with open_files(filenames) as input_files:
        output_filenames = []

        for size in sizes:
            if size == 0:
                output_filenames.append(None)
                continue

            with open_temp_files(num=len(filenames)) as output_files:
                for input_file, output_file in zip(input_files, output_files):
                    # if size is None, this will read the whole file,
                    # that's why we put train last
                    output_file.writelines(islice(input_file, size))
                output_filenames.append([f.name for f in output_files])

        return output_filenames


def create_subwords(filename, output_filename, size):
    cmd = ['scripts/learn_bpe.py', '--input', filename, '-s', str(size), '--output', output_filename]
    subprocess.call(cmd)


def apply_subwords(filename, bpe_filename):
    with open_temp_files(num=1) as output_:
        output_, = output_
        cmd = ['scripts/apply_bpe.py', '--input', filename, '--codes', bpe_filename]
        subprocess.call(cmd, stdout=output_)

        return output_.name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='training corpus')
    parser.add_argument('extensions', nargs='+', help='list of extensions '
                        '(first extension is the main source, '
                        'last extension is the target)')

    parser.add_argument('output_dir',
                        help='directory where the files will be copied')

    parser.add_argument('--mode', help='prepare: preprocess and copy corpora, '
                        'vocab: create vocabularies from train files, '
                        'all: do all of the above', default='all',
                        choices=('prepare', 'vocab', 'all'))

    parser.add_argument('--output-prefix', help='start filenames with '
                        'this prefix', default='')

    parser.add_argument('--suffix', default='train')
    parser.add_argument('--dev-suffix', default='dev')
    parser.add_argument('--test-suffix', default='test')

    # TODO: vocab prefix
    parser.add_argument('--dev-corpus', help='development corpus')
    parser.add_argument('--test-corpus', help='test corpus')

    parser.add_argument('--scripts', help='path to script directory', default='scripts')

    parser.add_argument('--dev-size', type=int,
                        help='size of development corpus', default=0)
    parser.add_argument('--test-size', type=int,
                        help='size of test corpus', default=0)
    parser.add_argument('--train-size', type=int,
                        help='size of training corpus (default: maximum)')

    parser.add_argument('--lang', nargs='+', help='optional list of language '
                                                  'codes (when different '\
                                                  'than file extensions)')
    parser.add_argument('--character-level', nargs='+', help='builds '
                        'a character-level vocabulary for the given extensions, '
                        'line length filtering is also performed at the '
                        'character level')
    parser.add_argument('--subwords', nargs='+', help='convert words to subword'
                        'units for the given extensions')
    parser.add_argument('--bpe-path', help='path to existing subword units (corpus prefix)')

    parser.add_argument('--normalize-punk', help='normalize punctuation',
                        action='store_true')
    parser.add_argument('--normalize-digits', help='normalize digits '
                        '(replace all digits with 0)', action='store_true')
    parser.add_argument('--lowercase', help='put everything to lowercase',
                        action='store_true')
    parser.add_argument('--shuffle', help='shuffle the corpus',
                        action='store_true')
    parser.add_argument('--no-tokenize', dest='tokenize',
                        help='no tokenization', action='store_false')
    parser.add_argument('--normalize-moses', help='remove | symbols '
                        '(used as delimiters by moses)', action='store_true')
    parser.add_argument('--remove-duplicates', help='remove duplicate pairs',
                        action='store_true')
    parser.add_argument('--escape-special-chars', help='escape special characters',
                        action='store_true')
    parser.add_argument('--remove-duplicate-lines', help='more restrictive '
                        'than --remove-duplicates, remove any pair of lines '
                        'whose source or target side was already seen.')
    parser.add_argument('-v', '--verbose', help='verbose mode',
                        action='store_true')

    parser.add_argument('--min', nargs='+', type=int, default=1,
                        help='min number of tokens per line')
    parser.add_argument('--max', nargs='+', type=int, default=50,
                        help='max number of tokens per line (0 for no limit)')
    parser.add_argument('--vocab-size', nargs='+', type=int, help='size of '
                        'the vocabularies', default=30000)
    parser.add_argument('--vocab-path', help='path to existing vocabularies (corpus prefix)')
    parser.add_argument('--threads', type=int, default=16)

    args = parser.parse_args()

    def fixed_length_arg(name, value, length):
        if value is None:
            return value
        elif isinstance(value, int):
            return [value for _ in range(length)]
        elif len(value) == 1:
            return [value[0] for _ in range(length)]
        elif len(value) == length:
            return value
        else:
            sys.exit('wrong number of values for parameter {}'.format(name))

    n = len(args.extensions)
    args.min = fixed_length_arg('--min', args.min, n)
    args.max = fixed_length_arg('--max', args.max, n)
    args.vocab_size = fixed_length_arg('--vocab-size', args.vocab_size, n)

    args.max = [i if i > 0 else float('inf') for i in args.max]

    assert not (args.subwords and args.character_level), 'error: --subwords and --character-level' \
                                                         'options are incompatible'

    if args.lang is None:
        args.lang = args.extensions
    elif len(args.lang) != n:
        sys.exit('wrong number of values for parameter --lang')
    if args.vocab_path is not None:
        args.vocab_path = ['{}.{}'.format(args.vocab_path, ext) for ext in args.extensions]
    if args.bpe_path is not None:
        args.bpe_path = ['{}.{}'.format(args.bpe_path, ext) for ext in args.extensions]

    if args.verbose:
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    create_corpus_ = args.mode in ('all', 'prepare')
    create_vocab_ = args.mode in ('all', 'vocab')

    if not os.path.exists(args.output_dir):
        logging.info('creating directory')
        os.makedirs(args.output_dir)

    input_corpora = (args.dev_corpus, args.test_corpus, args.corpus)

    # corpora names are a concatenation of a prefix and a suffix (e.g. europarl.train)
    output_corpora_names = [
        args.output_prefix if not suffix else
        suffix if not args.output_prefix else
        '{}.{}'.format(args.output_prefix, suffix)
        for suffix in (args.dev_suffix, args.test_suffix, args.suffix)
    ]

    # full paths to dev, test and train corpora
    output_corpora = [
        os.path.join(args.output_dir, name)
        for name in output_corpora_names
    ]

    try:
        # list of temporary files for each corpus (dev, test, train)
        # a value of None (default for dev and test) means that no
        # corpus is provided
        corpora = [
            (None if corpus is None else
            ['{}.{}'.format(corpus, ext) for ext in args.extensions])
            for corpus in input_corpora
        ]

        sizes = [
            args.dev_size if not args.dev_corpus else 0,
            args.test_size if not args.test_corpus else 0,
            args.train_size  # train must be last for `split_corpus`
        ]

        ## process corpora and copy them to their destination
        if create_corpus_:
            for i, corpus in enumerate(corpora):
                if corpus is not None:
                    corpora[i] = process_corpus(corpus, args)

            # split corpus into train/dev/test
            # size of 0: no corpus is created
            # size of None: copy everything (default for train)
            # if dev/test corpus is provided, we don't split
            if any(sizes):
                logging.info('splitting files')
                split_corpora = split_corpus(corpora[-1], sizes)

                # union of `filenames` and `split_filenames`
                for corpus, split_corpus in zip(corpora, split_corpora):
                    if split_corpus is not None:
                        corpus[:] = split_corpus

            # filter corpora by line length
            # TODO: character-level filtering
            for corpus in corpora:
                corpus[:] = filter_corpus(corpus, args)

            # create subwords and process files accordingly
            if args.subwords:
                if args.bpe_path:
                    bpe_filenames = args.bpe_path
                else:
                    bpe_filenames = [
                        os.path.join(args.output_dir, 'bpe.{}'.format(ext))
                        for ext in args.extensions
                    ]

                    # create subwords
                    train_corpus = corpora[-1]
                    for ext, filename, bpe_filename, size in zip(args.extensions, train_corpus, bpe_filenames,
                                                                 args.vocab_size):
                        if ext in args.subwords:
                            # this does not ensure a vocabulary size of `size`
                            create_subwords(filename, bpe_filename, size)

                # apply subwords to train, dev and test
                for corpus in corpora:
                    if corpus is None:
                        continue

                    filenames = [
                        apply_subwords(filename, bpe_filename) if ext in args.subwords else filename
                        for ext, filename, bpe_filename in zip(args.extensions, corpus, bpe_filenames)
                    ]

                    # filter lines by length again...
                    filenames = filter_corpus(filenames, args)
                    corpus[:] = filenames

            # move temporary files to their destination
            for i, corpus in enumerate(corpora):
                if corpus is None:
                    continue
                output_corpus = ['{}.{}'.format(output_corpora[i], ext)
                                 for ext in args.extensions]
                for filename, output_filename in zip(corpus, output_corpus):
                    shutil.move(filename, output_filename)

                corpora[i] = output_corpus

        ## create vocabularies
        vocab_output_filenames = [
            os.path.join(args.output_dir, 'vocab.{}'.format(ext))
            for ext in args.extensions
        ]

        if args.vocab_path is not None:
            logging.info('reading vocabulary files')
            vocabs = [read_vocabulary(filename) for filename in args.vocab_path]
            if create_vocab_:
                # copy vocabularies if necessary
                for vocab_filename, output_filename in zip(args.vocab_path,
                                                           vocab_output_filenames):
                    if vocab_filename != output_filename:
                        shutil.copy(vocab_filename, output_filename)
        elif create_vocab_:
            logging.info('creating vocabulary files')
            # training corpus is used to create vocabulary
            train_corpus = corpora[-1]

            vocabs = []
            for filename, output_filename, size, ext in zip(train_corpus,
                                                            vocab_output_filenames,
                                                            args.vocab_size,
                                                            args.extensions):
                character_level = args.character_level and ext in args.character_level
                vocab = create_vocabulary(filename, output_filename, size, character_level)
                vocabs.append(vocab)
        else:
            vocabs = None

    finally:
        logging.info('removing temporary files')
        for name in temporary_files:  # remove temporary files
            try:
                os.remove(name)
            except OSError:
                pass
