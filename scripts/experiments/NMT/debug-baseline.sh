#!/usr/bin/env bash

set -e  # exit script on failure

data_dir=data/debug_fr-en
train_dir=model/debug_fr-en
embedding_size=256
vocab_size=30000
layers=1
steps_per_checkpoint=20
steps_per_eval=100
dev_size=100
gpu_opts="--no-gpu --gpu-id 1"
subwords=

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
corpus_train=data/raw/news-commentary.fr-en
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
${subwords}

head -n ${dev_size} ${data_dir}/dev.ids.en > ${data_dir}/dev.${dev_size}.ids.en
head -n ${dev_size} ${data_dir}/dev.ids.fr > ${data_dir}/dev.${dev_size}.ids.fr
head -n ${dev_size} ${data_dir}/dev.en > ${data_dir}/dev.${dev_size}.en
head -n ${dev_size} ${data_dir}/dev.fr > ${data_dir}/dev.${dev_size}.fr
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
${gpu_opts} \
--steps-per-checkpoint ${steps_per_checkpoint} \
--steps-per-eval ${steps_per_eval} \
--dev-prefix dev.${dev_size}
