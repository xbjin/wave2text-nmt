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
$MOSES_DIR/bin/lmplz -o {order} < {corpus}.{extension} > {corpus}.arpa.{extension}
$MOSES_DIR/bin/build_binary {corpus}.arpa.{extension} {corpus}.blm.{extension}\
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('corpus', help='name of the corpus')
    parser.add_argument('extension', help='extension of the file (e.g. en or fr)')
    parser.add_argument('--order', default=3, type=int, help='order of the n-grams')

    args = parser.parse_args()

    commands = commands.strip().format(**vars(args))

    for cmd in commands.split('\n'):
        logging.info(cmd)
        os.system(cmd)
