#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import subprocess
import tempfile

help_msg = """\
Prepare a parallel corpus.

Usage example:
    prepare-data.py data/news fr en output/ --dev-corpus data/news-dev --max 0 --lowercase

This will create 4 files in output/: train.src, train.trg, dev.src, dev.trg
These files will be tokenized and lowercased, and empty lines will be filtered out.
"""

def process(filename, tokenize=False, lowercase=False):
    pass


def process_corpus(corpus, output_dir, args):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='training corpus')
    parser.add_argument('src_ext', help='source extension')
    parser.add_argument('trg_ext', help='target extension')
    parser.add_argument('output_dir', help='directory where the files will be copied')

    parser.add_argument('--dev-corpus', help='development corpus')
    parser.add_argument('--test-corpus', help='test corpus')
    parser.add_argument('--src-lang', help='source language code (when different than file extension)')
    parser.add_argument('--trg-lang', help='target language code (when different than file extension)')

    parser.add_argument('--normalize-punc', help='normalize punctuation')
    parser.add_argument('--lowercase', help='put everything to lowercase')
    parser.add_argument('--shuffle', help='shuffle the corpus')
    parser.add_argument('--no-tokenize', help='no tokenization')

    parser.add_argument('--min', type=int, default=1, help='min number of words in a line')
    parser.add_argument('--max', type=int, default=50, help='max number of words in a line (0 is no limit)')

    args = parser.parse_args()

    if args.src_lang is None:
        args.src_lang = args.src_ext
    if args.trg_lang is None:
        args.trg_lang = args.trg_ext

    if args.max == 0:
        args.max = float('inf')

