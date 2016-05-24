#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import izip, islice
from random import shuffle
from contextlib import contextmanager
import argparse
import subprocess
import tempfile
import os
import logging
import sys
import shutil


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

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"

_UNKS = ['_UNK'] + ['_UNK{0:+d}'.format(i) for i in range(-7, 8)]  # _UNK-7, ..., _UNK+0, ..., _UNK+7
_UNK = _UNKS[0]

_START_VOCAB_BASIC = [_PAD, _GO, _EOS, _UNK]
_START_VOCAB_UNK = [_PAD, _GO, _EOS] + _UNKS

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_IDs = range(3, 3 + len(_UNKS))
UNK_ID = UNK_IDs[0]

temporary_files = []


@contextmanager
def open_files(names, mode='r'):
    files = []
    try:
        for name_ in names:
            files.append(open(name_, mode=mode))
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


def create_vocabulary(filename, output_filename, size, unk_align=False):
    start_vocab = _START_VOCAB_UNK if unk_align else _START_VOCAB_BASIC

    logging.info('creating vocabulary {} from {}'.format(output_filename,
                                                         filename))
    vocab = {}
    with open(filename) as input_file, \
         open(output_filename, 'w') as output_file:
        for line in input_file:
            for w in line.split():
                vocab[w] = vocab.get(w, 0) + 1

        vocab_list = start_vocab + sorted(vocab, key=vocab.get, reverse=True)
        if 0 < size < len(vocab_list):
            vocab_list = vocab_list[:size]

        output_file.writelines(w + '\n' for w in vocab_list)

    return dict(map(reversed, enumerate(vocab_list)))

def append_unk_vocab(vocab, vocab_file):
    with open(vocab_file, 'w') as output_file:    
        vocab_list=_START_VOCAB_UNK+vocab[4:]    
        if 0 < size < len(vocab_list):
                vocab_list = vocab_list[:size]
    
        output_file.writelines(w + '\n' for w in vocab_list)    

def create_ids(filename, output_filename, vocab):
    with open(filename) as input_file, \
         open(output_filename, 'w') as output_file:
        for line in input_file:
            ids = [str(vocab.get(w, UNK_ID)) for w in line.split()]
            output_file.write(' '.join(ids) + '\n')


def create_ids_with_align(filename, output_filename, vocab, align_filename):
    with open_files([filename, align_filename]) as files, \
         open(output_filename, 'w') as output_file:
        for line, align_line in zip(*files):
            # reverse the alignment
            align_pair = dict(map(int, item.split('-'))[::-1]
                              for item in align_line.split())

            ids = []

            for trg_pos, trg_word in enumerate(line.split()):
                token = vocab.get(trg_word, None)
                if token is None:
                    src_pos = align_pair.get(trg_pos, trg_pos + 8)

                    offset = src_pos - trg_pos
                    if abs(offset) <= 7:
                        token = UNK_IDs[8 + offset]
                    else:
                        token = UNK_IDs[0]

                ids.append(str(token))
            output_file.write(' '.join(ids) + '\n')


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
        if args.normalize_moses:
            processes.append(['sed', 's/|//g'])

        if args.tokenize:
            processes.append([path_to('tokenizer.perl'), '-l', lang, '-threads',
                              str(args.threads)])
        if args.lowercase:
            processes.append([path_to('lowercase.perl')])
        if args.normalize_digits:
            processes.append(['sed', 's/[[:digit:]]/0/g'])


        ps = None

        for i, process in enumerate(processes):
            stdout = output_ if i == len(processes) - 1 else subprocess.PIPE
            stdin = input_ if i == 0 else ps.stdout

            ps = subprocess.Popen(process, stdin=stdin, stdout=stdout,
                                  stderr=open('/dev/null', 'w'))

        ps.wait()
        return output_.name


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
                    lines.append(lines_)
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


def create_align(filenames, args):
    with open_temp_files(num=1) as output_file, open_files(filenames) as files:
        output_file, = output_file
        for src_line, trg_line in izip(*files):
            output_file.write("{} ||| {}\n".format(src_line.strip(),
                                                   trg_line.strip()))
        tmp_filename = output_file.name

    with open_temp_files(num=1) as align_file:
        align_file, = align_file

        fast_align_bin = os.path.join(args.scripts, args.fast_align_bin)
        args_ = [fast_align_bin, '-i', tmp_filename, '-d', '-o', '-v',
                 '-I', str(args.fast_align_iter)]

        subprocess.call(args_, stdout=align_file, stderr=subprocess.PIPE)
        return align_file.name
                

def create_lookup_dictionary(filenames, align_filename, output_filename, args):
    counts = {}  # word pair counts
    with open_files(list(filenames) + [align_filename]) as files:
       for src_line, trg_line, alignment_line in izip(*files):
           alignment = [map(int, item.split('-'))
                        for item in alignment_line.split()]

           src_tokens = src_line.split()
           trg_tokens = trg_line.split()

           for src_id, trg_id in alignment:
               pair = src_tokens[src_id], trg_tokens[trg_id]
               counts[pair] = counts.get(pair, 0) + 1

    # keep only pairs with a count above given threshold
    frequent_pairs = dict((pair, count) for pair, count in counts.iteritems()
                          if count >= args.dict_threshold)

    # for each source word, find the target word with the highest count
    dictionary = {}
    for src_word, trg_word in sorted(frequent_pairs, key=frequent_pairs.get,
                                     reverse=True):
        dictionary.setdefault(src_word, trg_word)
    
    # write dict to file
    with open(output_filename, 'w') as output_file:
        for key, value in dictionary.iteritems():
            output_file.write(key + ' ' + value + '\n')

            
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
                        'ids: map words to ids using vocabulary, '
                        'all: do all of the above', default='all',
                        choices=('prepare', 'vocab', 'ids', 'all'))

    parser.add_argument('--output-prefix', help='start filenames with '
                        'this prefix', default='')

    parser.add_argument('--suffix', default='train')
    parser.add_argument('--dev-suffix', default='dev')
    parser.add_argument('--test-suffix', default='test')

    # TODO: vocab prefix, lookup dict name, align prefix, ids suffix

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
    parser.add_argument('--vocab-path', nargs='+', help='path to existing '
                        'vocabularies')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--unk-align', help='align target unknown words with the '
                        'source using special UNK symbols', action='store_true')
    parser.add_argument('--dict-threshold', help='min count of a word pair '
                        'in the dictionary', type=int, default=100)
    parser.add_argument('--fast-align-bin', help='name of the fast_align '
                        'binary (relative to script directory)',
                        default='fast_align')
    parser.add_argument('--fast-align-iter', help='number of iterations in '
                        'fast_align', type=int, default=5)

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

    if args.lang is None:
        args.lang = args.extensions
    elif len(args.lang) != n:
        sys.exit('wrong number of values for parameter --lang')
    if args.vocab_path is not None and len(args.vocab_path) != n:
        sys.exit('wrong number of values for parameter --vocab_path')

    if args.verbose:
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    create_corpus_ = args.mode in ('all', 'prepare')
    create_vocab_ = args.mode in ('all', 'vocab')
    create_ids_ = args.mode in ('all', 'ids')

    if create_ids_ and args.vocab_path is None:
        # vocabulary is needed for conversion to ids
        create_vocab_ = True

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

    # corpora names must be non-empty and unique
    if (not all(output_corpora_names) or
        len(set(output_corpora_names)) != len(output_corpora_names)):
        sys.exit('invalid output names')

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
                for i, corpus in enumerate(split_corpora):
                    if corpus is None:
                        continue
                    corpora[i] = corpus

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
                unk_align = args.unk_align and ext == args.extensions[-1]
                vocab = create_vocabulary(filename, output_filename, size, unk_align)
                vocabs.append(vocab)
        else:
            vocabs = None

        ## align and create ids
        if create_ids_:
            if args.unk_align:
                logging.info('creating alignment')
                # we only align training data
                train_corpus = corpora[-1]

                # alignment only works for a pair of languages
                # by convention, first extension is the main source
                # and last extension is the target
                align_input_filenames = (train_corpus[0], train_corpus[-1])

                # TODO: possibility to use a different corpus to create lookup dict
                align_filename = create_align(align_input_filenames, args)
                # use the newly created alignment to build a dictionary
                logging.info('creating lookup dictionary')
                dict_filename = os.path.join(args.output_dir, 'lookup_dict')
                create_lookup_dictionary(align_input_filenames,
                                         align_filename, dict_filename, args)
            else:
                align_filename = None

            logging.info('creating ids')
            for corpus, output_corpus, size in zip(corpora, output_corpora, sizes):
                if corpus is None:
                    continue
                for ext, vocab, filename in zip(args.extensions, vocabs, corpus):
                    output_filename = '{}.ids.{}'.format(output_corpus, ext)

                    # special UNK symbols for target train file
                    if (args.unk_align and ext == args.extensions[-1] and
                        output_corpus == output_corpora[-1]):
                        #if external vocab is given we append special unk tokens after removing old special tokens
                        if args.vocab_path is not None:  
                            append_unk_vocab(vocab, args.vocab_path[-1])
                        create_ids_with_align(filename, output_filename, vocab, align_filename)
                    else:
                        create_ids(filename, output_filename, vocab)

    finally:
        logging.info('removing temporary files')
        for name in temporary_files:  # remove temporary files
            try:
                os.remove(name)
            except OSError:
                pass
