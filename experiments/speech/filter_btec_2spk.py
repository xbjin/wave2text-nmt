#!/usr/bin/python2

"""
Pre-processing file for BTEC spoken data (by Laurent & Margaux)
"""

import wave
import os
import shutil
import struct
import random

data_dir = 'data/btec'
speaker = 'Laurent'
txt_filename = os.path.join(data_dir, 'btec.{}.en'.format(speaker))
feats_filename = os.path.join(data_dir, 'btec.{}.feats41'.format(speaker))

test_size = 400

train_file = os.path.join(data_dir, 'train.en')
test_files = [os.path.join(data_dir, 'dev.en'),
              os.path.join(data_dir, 'test1.en'),
              os.path.join(data_dir, 'test2.en')]

test_txt_output = os.path.join(data_dir, 'test.{}.en'.format(speaker))
test_feats_output = os.path.join(data_dir, 'test.{}.feats41'.format(speaker))
train_txt_output = os.path.join(data_dir, 'train.{}.en'.format(speaker))
train_feats_output = os.path.join(data_dir, 'train.{}.feats41'.format(speaker))

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

    data = struct.pack('i' + 'f' * n, frames, *feats)
    if line in train_lines and line in test_lines:
      l = lines_in_both
    elif line in train_lines:
      l = lines_in_train
    elif line in test_lines:
      l = lines_in_test
    else:
      l = lines_in_neither

    l.append((line, data))

total = sum(map(len, [lines_in_train, lines_in_test, lines_in_both, lines_in_neither]))
print('{}/{} lines in both train and test (discarded)'.format(len(lines_in_both), total))
print('{}/{} lines in train'.format(len(lines_in_train), total))
print('{}/{} lines in test'.format(len(lines_in_test), total))
print('{}/{} lines in neither'.format(len(lines_in_neither), total))


i = max(0, 400 - len(lines_in_train))

g = random.Random(42)
g.shuffle(lines_in_neither)

write_to_test = lines_in_train + lines_in_neither[:i]
write_to_train = lines_in_test + lines_in_neither[i:]

with open(test_txt_output, 'w') as test_txt_file, open(train_txt_output, 'w') as train_txt_file:
  with open(test_feats_output, 'wb') as test_feats_file, open(train_feats_output, 'wb') as train_feats_file:
    test_feats_file.write(struct.pack('ii', len(write_to_test), dim))
    train_feats_file.write(struct.pack('ii', len(write_to_train), dim))

    for line, feats in write_to_test:
      test_feats_file.write(feats)
      test_txt_file.write(line)
    for line, feats in write_to_train:
      train_feats_file.write(feats)
      train_txt_file.write(line)
