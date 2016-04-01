#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import subprocess
import tempfile
import os

help_msg = """\
Prepare a parallel corpus.

Usage example:
    prepare-data.py data/news fr en output/ --dev-corpus data/news-dev --max 0 --lowercase

This will create 4 files in output/: train.src, train.trg, dev.src, dev.trg
These files will be tokenized and lowercased, and empty lines will be filtered out.
"""


def tokenize(filename, lang, threads):
    with open(filename) as input_,\
         tempfile.NamedTemporaryFile('w', delete=False) as output_:
        subprocess.call(['scripts/tokenizer.perl', '-l', lang, '-threads',
                         str(threads)], stdin=input_, stdout=output_)
        return output_.name


def process_corpus(corpus, args):
    if corpus is None:
        return

    filenames = ['{}.{}'.format(corpus, ext) for ext in args.extensions]

    # tokenization
    delete = False
    if not args.no_tokenize:
        filenames = [tokenize(filename, lang, args.threads) for filename, lang
                     in zip(filenames, args.lang)]
        delete = True

    if delete:
        # remove temporary files
        for filename in filenames:
            os.remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='training corpus')
    parser.add_argument('extensions', nargs='+', help='list of extensions')

    parser.add_argument('output_dir', help='directory where the files will be copied')

    parser.add_argument('--dev-corpus', help='development corpus')
    parser.add_argument('--test-corpus', help='test corpus')

    parser.add_argument('--dev-size', type=int, help='size of development corpus')
    parser.add_argument('--test-size', type=int, help='size of test corpus')
    parser.add_argument('--train-size', type=int, help='size of training corpus')

    parser.add_argument('--lang', nargs='+', help='list of language codes'
                                                  '(when different than file '\
                                                  'extensions)')

    parser.add_argument('--normalize', help='normalize punctuation')
    parser.add_argument('--normalize-numbers', help='normalize numbers')
    parser.add_argument('--lowercase', help='put everything to lowercase')
    parser.add_argument('--shuffle', help='shuffle the corpus')
    parser.add_argument('--no-tokenize', help='no tokenization')

    parser.add_argument('--min', nargs='+', type=int, default=1,
                        help='min number of words in a line')
    parser.add_argument('--max', nargs='+', type=int, default=50,
                        help='max number of words in a line (0 for no limit)')

    parser.add_argument('--vocab-size', nargs='+', type=int, default=None)

    parser.add_argument('--threads', type=int, default=16)

    args = parser.parse_args()

    if args.src_lang is None:
        args.src_lang = args.src_ext
    if args.trg_lang is None:
        args.trg_lang = args.trg_ext

    if args.max == 0:
        args.max = float('inf')

    process_corpus(args.corpus, args)
    process_corpus(args.dev_corpus, args)


    # normalize-punctuation.perl | tokenizer.perl | lowercase.perl | normalize-numbers > filename
    # clean corpus0 corpus1
    # shuffle corpus1 corpus2
    # split corpus2 train test dev

    # create-vocab train vocab
    # create-ids train vocab ids