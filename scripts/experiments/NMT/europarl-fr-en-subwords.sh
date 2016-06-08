#!/usr/bin/env bash

set -e  # exit script on failure

# set as variable before running script
if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

data_dir=data/europarl_fr-en_subwords
train_dir=model/europarl_fr-en_subwords
embedding_size=512
vocab_size=30000
layers=1

mkdir -p ${train_dir}
mkdir -p ${data_dir}

if test "$(ls -A "${train_dir}")"; then
    read -p "warning: train dir is not empty, continue? [y/N] " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

if test "$(ls -A "${data_dir}")"; then
    echo "warning: data dir is not empty, skipping data preparation"
else
echo "### downloading data"

# assume that data is in data/raw, until we fix fetch-corpus.py
corpus_train=data/raw/europarl.fr-en
corpus_dev=data/raw/news-dev.fr-en
corpus_test=data/raw/news-test.fr-en

echo "### pre-processing data"

./scripts/prepare-data.py ${corpus_train} fr en ${data_dir} --mode all \
--verbose \
--normalize-digits \
--normalize-punk \
--max 50 \
--vocab-size ${vocab_size} \
--dev-corpus ${corpus_dev} \
--test-corpus ${corpus_test} \
--subwords

head -n1000 ${data_dir}/dev.ids.en > ${data_dir}/dev.1000.ids.en
head -n1000 ${data_dir}/dev.ids.fr > ${data_dir}/dev.1000.ids.fr

head -n1000 ${data_dir}/dev.en > ${data_dir}/dev.1000.en
head -n1000 ${data_dir}/dev.fr > ${data_dir}/dev.1000.fr
fi

echo "### training model"

python -m translate ${data_dir} ${train_dir} \
--train \
--size ${embedding_size} \
--num-layers ${layers} \
--src-vocab-size ${vocab_size} \
--trg-vocab-size ${vocab_size} \
--src-ext fr \
--trg-ext en \
--verbose \
--log-file ${train_dir}/log.txt \
--gpu-id ${GPU} \
--steps-per-checkpoint 1000 \
--steps-per-eval 4000 \
--dev-prefix dev.1000
