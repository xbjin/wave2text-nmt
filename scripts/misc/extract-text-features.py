#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import numpy as np
import struct
import tensorflow as tf
from translate.utils import initialize_vocabulary, sentence_to_token_ids

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='input text file')
parser.add_argument('--checkpoint', help='checkpoint file (containing the embeddings)', required=True)
parser.add_argument('--vocab', help='vocab file corresponding to this checkpoint', required=True)
parser.add_argument('--vocab-size', type=int, required=True)
parser.add_argument('--embedding-size', type=int, required=True)
parser.add_argument('output_file', help='output file')

args = parser.parse_args()

# read embeddings from checkpoint
reader = tf.train.NewCheckpointReader(args.checkpoint)

with tf.device('/cpu:0'):
  shape = args.vocab_size, args.embedding_size
  embeddings = reader.get_tensor('multi_encoder/encoder_fr/embedding')

vocab = initialize_vocabulary(args.vocab)

with open(args.filename) as input_file, open(args.output_file, 'wb') as output_file:
  lines = list(input_file)
  output_file.write(struct.pack('ii', len(lines), args.embedding_size))

  for line in lines:
    token_ids = sentence_to_token_ids(line, vocab.vocab)
    feats = [embeddings[id_] for id_ in token_ids]
    length = len(feats)

    feats = np.concatenate(feats)
    output_file.write(struct.pack('i' + 'f' * len(feats), length, *feats))
