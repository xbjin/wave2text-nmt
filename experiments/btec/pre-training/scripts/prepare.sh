#!/usr/bin/env bash

news_corpus=data/raw/news-commentary.fr-en
btec_corpus=data/raw/btec.fr-en/btec-train
dev_corpus=data/raw/btec.fr-en/btec-dev-concat
test_corpus=data/raw/btec.fr-en/btec-test

output_dir=experiments/btec/pre-training/data

scripts/prepare-data.py ${news_corpus} fr en ${output_dir} --output news --mode prepare \
--lowercase --max 0 --min 1 --threads 4 --normalize-punk
scripts/prepare-data.py ${btec_corpus} fr en ${output_dir} --output btec --mode prepare \
--lowercase --max 0 --min 1 --threads 4

scripts/prepare-data.py ${output_dir}/btec fr en ${output_dir} --mode vocab

scripts/prepare-data.py ${dev_corpus} fr en mref.en ${output_dir} --output dev --mode prepare \
--lowercase --max 0 --min 1 --threads 4
scripts/prepare-data.py ${test_corpus}1 fr en mref.en ${output_dir} --output test1 --mode prepare \
--lowercase --max 0 --min 1 --threads 4
scripts/prepare-data.py ${test_corpus}2 fr en mref.en ${output_dir} --output test2 --mode prepare \
--lowercase --max 0 --min 1 --threads 4