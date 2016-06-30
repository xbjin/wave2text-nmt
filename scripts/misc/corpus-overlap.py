#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import argparse
from contextlib import contextmanager
from itertools import izip


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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='compute overlap between two corpora')

  parser.add_argument('first_corpus', help='first corpus')
  parser.add_argument('second_corpus', help='second corpus')
  parser.add_argument('extensions', nargs='+', help='list of extensions')
  parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')

  args = parser.parse_args()

  filenames1 = ['{}.{}'.format(args.first_corpus, ext) for ext in args.extensions]
  filenames2 = ['{}.{}'.format(args.second_corpus, ext) for ext in args.extensions]

  with open_files(filenames1) as files:
    lines1_set = set(izip(*files))

    vocabs1 = [set(word for line in lines_ for word in line.split()) for lines_ in izip(*lines1_set)]

  with open_files(filenames2) as files:
    lines2 = list(izip(*files))
    lines2_set = set(lines2)

    words2 = [[word for line in lines_ for word in line.split()] for lines_ in izip(*lines2)]
    vocabs2 = [set(word for line in lines_ for word in line.split()) for lines_ in izip(*lines2_set)]

  line_overlap = sum(1 for line in lines2 if line in lines1_set)
  unique_line_overlap = sum(1 for line in lines2_set if line in lines1_set)

  print('{}/{} lines ({:.1%})'.format(line_overlap, len(lines2), line_overlap / len(lines2)))
  print('{}/{} unique lines ({:.1%})'.format(unique_line_overlap, len(lines2_set),
                                               unique_line_overlap / len(lines2_set)))

  print('data coverage')
  for vocab1, vocab2, words_, ext in zip(vocabs1, vocabs2, words2, args.extensions):
    intersection = vocab1.intersection(vocab2)
    print('{}: vocab {}/{} ({:.1%})'.format(ext, len(intersection), len(vocab2), len(intersection) / len(vocab2)))
    if args.verbose:
      print('unknown tokens: {}'.format(' '.join(vocab2 - intersection)))

    total_word_coverage = sum(1 for word in words_ if word in vocab1)
    print('    total {}/{} ({:.1%})'.format(total_word_coverage, len(words_), total_word_coverage / len(words_)))