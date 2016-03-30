#!/usr/bin/env python2
from __future__ import division
import os
import argparse


help_msg = """\
Splits a text file into N files of identical size.

Usage example:
    split-n.py data/my_file.txt data/split_dir 4

Say `my_file.txt` has 1000 lines, this will create 4 files:
split_dir/0, split_dir/1, split_dir/2, split_dir/2 of 250 lines each.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('file', help='path of the file to split')
    parser.add_argument('output_dir', help='destination directory of the split parts')
    parser.add_argument('n_splits', type=int, help='number of splits')

    args = parser.parse_args()
    dataset, output_dir, n_splits = args.file, args.output_dir, args.n_splits

    total_lines = sum(1 for _ in open(dataset))
    lines_per_split = total_lines // n_splits
    input_file = open(dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(n_splits):
        lines = lines_per_split
        if i == n_splits - 1:
            lines += total_lines % n_splits

        filename = os.path.join(output_dir, str(i))
        with open(filename, 'w') as output_file:
            for _ in range(lines):
                output_file.write(next(input_file))
