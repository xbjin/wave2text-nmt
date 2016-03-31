#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import argparse

help_msg = """\
Removes too long or too short lines from given tokenized corpus. Tokens must
be separated by a single whitespace.

Usage example:
    clean_corpus.py data/my_corpus data/my_corpus.clean --min 5 --max 80 fr en\
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='name of the input corpus (path without extension, e.g. data/my_corpus)')
    parser.add_argument('output', help='name of the output corpus (must be different than input corpus)')
    parser.add_argument('--min', type=int, default=1, help='min number of tokens')
    parser.add_argument('--max', type=int, default=50, help='max number of tokens (0 for no limit)')
    parser.add_argument('extensions', nargs='+', help='extensions (e.g. fr, en)')

    args = parser.parse_args()

    min_, max_ = args.min, args.max
    extensions = args.extensions
    corpus = args.corpus
    output = args.output

    print args

    assert corpus != output, 'input and output corpora must be different'

    input_files = [open(corpus + '.' + ext) for ext in extensions]
    output_files = [open(output + '.' + ext, 'w') for ext in extensions]

    if max_ <= 0:
        max_ = sys.maxint

    try:
        while True:
            lines = [next(f) for f in input_files]
            if all(min_ <= len(line.split()) <= max_ for line in lines):
                for line, f in zip(lines, output_files):
                    f.write(line)
    except StopIteration:
        pass

    for f in output_files:
        f.close()
