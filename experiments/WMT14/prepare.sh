#!/usr/bin/env bash

raw_data_dir=data/raw
data_dir=experiments/WMT14/data

mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data_dir}/WMT14.fr-en fr en --no-tokenize --dev-corpus ${raw_data_dir}/ntst1213 \
--test-corpus ${raw_data_dir}/ntst14 --subwords --vocab-size 30000 --escape-special-chars --normalize-punk