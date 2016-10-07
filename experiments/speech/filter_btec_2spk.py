#!/usr/bin/python2

"""
Pre-processing file for BTEC spoken data (by Laurent & Margaux)
"""

import wave
import os
import shutil
import struct

txt_filename = 'data/raw/btec.fr-en/btec-Margaux.en'
feats_filename = 'data/raw/btec.fr-en/btec-Margaux.feats41'
test_size = 400

train_file = 'data/raw/btec.fr-en/train.en'
test_files = ['data/raw/btec.fr-en/dev.en', 'data/raw/btec.fr-en/test1.en', 'data/raw/btec.fr-en/test2.en']

test_txt_output = 'data/raw/btec.fr-en/test-Margaux.en'
test_feats_output = 'data/raw/btec.fr-en/test-Margaux.feats41'
train_txt_output = 'data/raw/btec.fr-en/train-Margaux.en'
train_feats_output = 'data/raw/btec.fr-en/train-Margaux.feats41'

train_lines = set(line for line in open(train_file))
test_lines = set(line for filename in test_files for line in open(filename))

lines_in_train = []
lines_in_test = []
lines_in_both = []
lines_in_neither = []

with open(txt_filename) as txt_file, open(feats_filename, 'rb') as feats_file:
  lines, dim = struct.unpack('ii', feats_file.read(8))

  for line in txt_file:
    frames, = struct.unpack('i', feats_file.read(4))
    n = frames * dim
    feats = struct.unpack('f' * n, feats_file.read(4 * n))

    if line in train_lines and line in test_lines:
      l = lines_in_both
    elif line in train_lines:
      l = lines_in_train
    elif line in test_lines:
      l = lines_in_test
    else:
      l = lines_in_neither

    l.append((line, feats))

total = sum(map(len, [lines_in_train, lines_in_test, lines_in_both, lines_in_neither]))
print('{}/{} lines in both train and test (discarded)'.format(len(lines_in_both), total))
print('{}/{} lines in train'.format(len(lines_in_train), total))
print('{}/{} lines in test'.format(len(lines_in_test), total))
print('{}/{} lines in neither'.format(len(lines_in_neither), total))


i = max(0, 400 - len(lines_in_train))
write_to_test = lines_in_train + lines_in_neither[:i]
write_to_train = lines_in_test + lines_in_neither[i:]

with open(test_txt_output, 'w') as test_txt_file, open(train_txt_output, 'w') as train_txt_file:
  with open(test_feats_output, 'wb') as test_feats_file, open(train_feats_output, 'wb') as train_feats_file:
    test_feats_file.write(struct.pack('ii', len(write_to_test), dim))
    train_feats_file.write(struct.pack('ii', len(write_to_train), dim))

    # TODO: write feats
    for line, feats in write_to_test:
      test_txt_file.write(line)
    for line, feats in write_to_train:
      train_txt_file.write(line)
