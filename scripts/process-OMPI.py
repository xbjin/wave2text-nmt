#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from contextlib import contextmanager
import argparse


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


parser = argparse.ArgumentParser()
parser.add_argument('post_edit_file', help='OMPI file containing the post-edits')
parser.add_argument('output_corpus', help='name of the output corpus')
parser.add_argument('--src-ext', default='fr')
parser.add_argument('--trg-ext', default='en')

args = parser.parse_args()

output_filenames = ['{}.{}'.format(args.output_corpus, ext) for ext in (args.src_ext, 'mt', args.trg_ext)]

with open(args.post_edit_file) as input_file, open_files(output_filenames, 'w') as output_files:
    for line in input_file:
        fields = line.split('\t')
        src, mt, pe = fields[3:6]
        sentences = fields[3:6]
        for output_file, sentence in zip(output_files, sentences):
            output_file.write(sentence + '\n')
