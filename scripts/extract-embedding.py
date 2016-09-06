#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse

"""
Extract embeddings from NMT model and save them in the word2vec format.
These embedding files can then be used to initialize a model, thanks to the `load_embeddings` parameter.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='TensorFlow checkpoint')
parser.add_argument('--name', required=True, help='name of the embedding parameter (e.g., fr)')
parser.add_argument('--vocab', required=True, help='path to vocabulary file')
parser.add_argument('--size', type=int, required=True, help='vector size')
parser.add_argument('--vocab-size', type=int, help='vocabulary size')

args = parser.parse_args()

reader = tf.train.NewCheckpointReader(args.checkpoint)

with open(args.vocab) as f:
  vocab = [line.rstrip('\r\n') for line in f]

with tf.device('/cpu:0'):
  embeddings = reader.get_tensor('embedding_{}'.format(args.name))

vocab_size = args.vocab_size or len(vocab)
if vocab_size != len(vocab):
  print('warning: inconsistent vocabulary size')

embeddings = embeddings.reshape((vocab_size, args.size))

print(vocab_size, args.size)
for word, vector in zip(vocab, embeddings):
  print(word, ' '.join(map(str, vector)))
