#!/usr/bin/env bash

set -e  # exit script on failure

# set as variable before running script
if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

data_dir=data/news-commentary_fr-en
train_dir=model/news-commentary_fr-en
gpu_id=${GPU}
embedding_size=1024
vocab_size=40000
num_samples=512
layers=3

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

corpus_train=data/raw/news-commentary.fr-en
corpus_dev=data/raw/ntst1213.fr-en
corpus_test=data/raw/ntst14.fr-en

echo "### pre-processing data"

./scripts/prepare-data.py ${corpus_train} fr en ${data_dir} --mode all \
--verbose \
--max 50 \
--vocab-size ${vocab_size}

./scripts/prepare-data.py ${corpus_dev} fr en ${data_dir} --mode all \
--suffix dev \
--verbose \
--no-tokenize \
--max 50 \
--vocab-path ${data_dir}/vocab

./scripts/prepare-data.py ${corpus_test} fr en ${data_dir} --mode prepare \
--suffix test \
--verbose \
--no-tokenize \
--max 50 \
--vocab-path ${data_dir}/vocab

head -n2000 ${data_dir}/dev.ids.en > ${data_dir}/dev.2000.ids.en
head -n2000 ${data_dir}/dev.ids.fr > ${data_dir}/dev.2000.ids.fr
head -n2000 ${data_dir}/dev.en > ${data_dir}/dev.2000.en
head -n2000 ${data_dir}/dev.fr > ${data_dir}/dev.2000.fr
fi

echo "### training model"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"
python -m translate ${data_dir} ${train_dir} \
--train \
--size ${embedding_size} \
--num-layers ${layers} \
--vocab-size ${vocab_size} \
--src-ext fr \
--trg-ext en \
--verbose \
--log-file ${train_dir}/log.txt \
--gpu-id ${GPU} \
--steps-per-checkpoint 1000 \
--steps-per-eval 4000 \
--dev-prefix dev.2000 \
--allow-growth \
--beam-size 1   # for fast eval during training
