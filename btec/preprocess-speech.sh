#!/usr/bin/env bash

data_dir=data/btec_speech
vocab_size=10000

mkdir -p ${data_dir}

if test "$(ls -A "${data_dir}")"; then
    echo "warning: data dir is not empty, skipping data preparation"
else

corpus_train=data/raw/btec.fr-en
corpus_dev=data/raw/btec-dev.fr-en
corpus_test=data/raw/btec-test.fr-en

echo "### pre-processing data"

# word-level output (try word-level and subwords)
./scripts/prepare-data.py ${corpus_train} fr en ${data_dir} \
--verbose \
--lowercase \
--dev-corpus ${corpus_dev} \
--test-corpus ${corpus_test} \
--vocab-size ${vocab_size}
fi

cp data/raw/btec.feats ${data_dir}/train.feats
cp data/raw/btec-dev.feats ${data_dir}/dev.feats
cp data/raw/btec-test.feats ${data_dir}/test.feats
