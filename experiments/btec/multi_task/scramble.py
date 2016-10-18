#!/usr/bin/python2
import sys
import random

for line in sys.stdin:
  tokens = line.split()
  random.shuffle(tokens)
  print ' '.join(tokens)
