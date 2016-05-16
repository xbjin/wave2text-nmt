#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
import logging
import argparse

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

help_msg = """Train a language model using Moses."""

commands = """\
$MOSES_DIR/bin/lmplz -o {order} < {corpus}.{extension} > {output}.arpa.{extension}
$MOSES_DIR/bin/build_binary {output}.arpa.{extension} {output}.blm.{extension}\
"""

if __name__ == '__main__':
    if 'MOSES_DIR' not in os.environ:
        sys.exit('Environment variable not defined')

    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('corpus', help='name of the corpus')
    parser.add_argument('extension', help='extension of the file (e.g. en or fr)')
    parser.add_argument('--output', help='output corpus if different than corpus')
    parser.add_argument('--order', default=3, type=int, help='order of the n-grams')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.corpus

    commands = commands.strip().format(**vars(args))

    for cmd in commands.split('\n'):
        logging.info(cmd)
        os.system(cmd)
