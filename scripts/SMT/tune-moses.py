#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
import logging
import argparse

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

help_msg = """Tune a given translation model trained with Moses."""

commands = """\
$MOSES_DIR/scripts/training/mert-moses.pl {corpus}.{src_ext} {corpus}.{trg_ext} \
$MOSES_DIR/bin/moses {config} --mertdir $MOSES_DIR/bin/ \
--decoder-flags="-threads {threads}" &> {log_file}
mv mert-work/moses.ini {config}.tuned
rm -rf mert-work\
"""

if __name__ == '__main__':
    if 'MOSES_DIR' not in os.environ:
        sys.exit('Environment variable not defined')

    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('corpus')
    parser.add_argument('src_ext')
    parser.add_argument('trg_ext')
    parser.add_argument('log_file')
    parser.add_argument('--threads', type=int, default=16)

    args = parser.parse_args()

    commands = commands.strip().format(**vars(args))

    for cmd in commands.split('\n'):
        logging.info(cmd)
        os.system(cmd)
