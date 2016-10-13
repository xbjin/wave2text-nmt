#!/usr/bin/env bash

srilm=bin/srilm/bin/i686-m64
data_dir=experiments/speech/data
order=3

mkdir -p ${data_dir}

# SRILM
${srilm}/ngram-count -text ${data_dir}/train.en -lm ${data_dir}/btec.arpa -order ${order}
