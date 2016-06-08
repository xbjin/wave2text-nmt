#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse


help_msg = """\
Filter a list of embeddings according to given vocabulary.\
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg)
    parser.add_argument('embedding_file')
    parser.add_argument('vocab_file')
    parser.add_argument('output_file')
    
    args = parser.parse_args()
    
    with open(args.embedding_file) as embedding_file, \
         open(args.vocab_file) as vocab_file, \
         open(args.output_file, 'w') as output_file:
        count, dimension = next(embedding_file).split()
        vocab = set(line.strip() for line in vocab_file)
        
        lines = []
        
        for line in embedding_file:
            word, vec = line.split(' ', 1)
            if word in vocab:
                lines.append('{} {}'.format(word, vec))
                
        count = len(lines)
        output_file.write('{} {}\n'.format(count, dimension))
        output_file.writelines(lines)

