#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import islice
import argparse

help_msg = """\
Does a dev/test/train split of a corpus.

Usage example:
    split-corpus.py data/my_corpus data/my_corpus --dev-size 1000 --test-size 1000 fr en
This will create 6 files: my_corpus.train.fr, my_corpus.train.en, my_corpus.dev.fr, etc.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='name of the input corpus (path without extension, e.g. data/my_corpus)')
    parser.add_argument('output', help='name of the output corpus')
    parser.add_argument('--dev-size', type=int, default=3000, help='size of dev corpus')
    parser.add_argument('--test-size', type=int, default=6000, help='size of test corpus')
    parser.add_argument('extensions', nargs='+', help='extensions (e.g. fr, en)')

    args = parser.parse_args()

    dev_size = args.dev_size
    test_size = args.test_size
    extensions = args.extensions
    corpus = args.corpus
    output = args.output

    input_files = [open(corpus + '.' + ext) for ext in extensions]
    output_files = [
        [open('{0}.{1}.{2}'.format(output, split_name, ext), 'w') for split_name in ['dev', 'test', 'train']]
        for ext in extensions
    ]
    sizes = [dev_size, test_size, None]

    for input, outputs in zip(input_files, output_files):
        for output, size in zip(outputs, sizes):
            output.writelines(islice(input, size))

    for f in sum(output_files, []):
        f.close()
