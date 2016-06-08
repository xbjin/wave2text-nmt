#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import subprocess
import sys
import os
import logging
import argparse

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

help_msg = """Spawn a number of Moses jobs to translate files created with the `split-n.py` script."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='path to moses.ini')
    parser.add_argument('dir', help='directory containing the file splits')
    parser.add_argument('from_', type=int, help='name of the starting split (e.g., 0 or 15)')
    parser.add_argument('jobs', type=int, help='number of processes to spawn')
    args = parser.parse_args()

    cfg = args.config
    directory = args.dir
    i = args.from_
    jobs = args.jobs

    processes = []
    for j in range(jobs):
        input_filename = os.path.join(directory, str(i + j))
        output_filename = os.path.join(directory, str(i + j) + '.out')

        cmd = '$MOSES_DIR/bin/moses -f {0} -threads 1 < {1} > {2} 2> /dev/null'.format(cfg, input_filename, output_filename)
        logging.info(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()
