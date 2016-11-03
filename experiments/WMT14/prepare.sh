#!/usr/bin/env bash

# NMT model using filtered WMT14 data, available on http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/

raw_data_dir=data/raw
data_dir=experiments/WMT14/data

mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data_dir}/WMT14.fr-en fr en ${data_dir} --no-tokenize \
--dev-corpus ${raw_data_dir}/ntst1213.fr-en \
--test-corpus ${raw_data_dir}/ntst14.fr-en \
--subwords --vocab-size 30000 --unescape-special-chars --normalize-punk --max 50 --bpe-path ${data_dir}/bpe

# 6000 is way too much
head -n2000 ${data_dir}/dev.fr > ${data_dir}/dev.2000.fr
head -n2000 ${data_dir}/dev.en > ${data_dir}/dev.2000.en
