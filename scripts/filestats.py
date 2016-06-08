#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import argparse
from collections import Counter

help_msg = """\
Provides basic information about a text file: number of words per line, etc.\
"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=help_msg,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('filename', help='input file path')
  args = parser.parse_args()

  filename = args.filename

  with open(filename) as f:
    lines = list(f)
    line_count = len(lines)
    lengths = Counter(len(line.split()) for line in lines)

    values = lengths.keys()
    word_count = sum(k * v for k, v in lengths.iteritems())

    avg = word_count / line_count

    print('lines={}, words={}'.format(line_count, word_count))
    print('Length distribution:', lengths)
    print('min={}, max={}, avg={}'.format(min(values), max(values), avg))
