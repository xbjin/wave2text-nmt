#!/usr/bin/env python3

import argparse
import subprocess
import os

"""
Proxy for `score.rb` scoring script.
Handles the multi-reference setting by default, and files are
assumed to be tokenized.
"""

parser = argparse.ArgumentParser()
parser.add_argument('hyp', help='path to hypothesis file')
parser.add_argument('ref', nargs='+', help='path to one or several reference files')

parser.add_argument('--no-case', dest='cased', action='store_false', help='case insensitive scores')
parser.add_argument('--script-path', default='bin/scoring/score.rb', help='path to scoring script')

if __name__ == '__main__':
    args = parser.parse_args()

    parameters = [args.script_path, '--hyp', args.hyp]
    if len(args.ref) == 1:
        parameters += ['--refs-laced', args.ref[0]]
    else:
        parameters += ['--refs'] + args.ref

    parameters += ['--no-norm', '--print', '--delete-results']
    if args.cased:
        parameters += ['--cased']

    devnull = open(os.devnull, 'w')
    output = subprocess.check_output(parameters, stderr=devnull).decode()
    scores = output.split()

    print('BLEU={} NIST={} TER={} RATIO={}'.format(*scores))
