#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
import logging
import argparse

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

help_msg = """Split a file into a number of files of equal size (in terms of lines)."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('file', help='file to split')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('n_splits', type=int, help='number of splits')
    args = parser.parse_args()

    dataset = args.dataset
    n_splits = args.n_splits
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        logging.info('creating directory {0}'.format(output_dir))
        os.makedirs(output_dir)

    total_lines = sum(1 for _ in open(dataset))
    lines_per_split = total_lines // n_splits
    input_file = open(dataset)

    for i in range(n_splits):
        lines = lines_per_split
        if i == n_splits - 1:
            lines += total_lines % n_splits

        filename = os.path.join(output_dir, str(i))
        with open(filename, 'w') as output_file:
            for _ in range(lines):
                output_file.write(next(input_file))
