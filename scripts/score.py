#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument('hyp')
parser.add_argument('ref', nargs='+')

parser.add_argument('--no-case', dest='cased', action='store_false')
parser.add_argument('--script-path', default='bin/scoring/score.rb')


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
  output = subprocess.check_output(parameters, stderr=devnull)
  scores = output.split()

  print 'BLEU={} NIST={} TER={} RATIO={}'.format(*scores)
