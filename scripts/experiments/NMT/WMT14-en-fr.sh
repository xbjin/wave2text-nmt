#!/usr/bin/env bash

set -e  # exit script on failure

# set as variable before running script
if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

data_dir=data/WMT14_en-fr_big
train_dir=model/WMT14_en-fr_big
gpu_id=${GPU}
embedding_size=1024
vocab_size=40000
num_samples=512
layers=2

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

corpus_train=data/raw/WMT14.fr-en
corpus_dev=data/raw/ntst1213.fr-en
corpus_test=data/raw/ntst14.fr-en

echo "### pre-processing data"

# no special preprocessing, and corpus is already tokenized
./scripts/prepare-data.py ${corpus_train} en fr ${data_dir} --mode all \
--verbose \
--no-tokenize \
--max 50 \
--dev-corpus ${corpus_dev} \
--test-corpus ${corpus_test} \
--vocab-size ${vocab_size}

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
--src-ext en \
--trg-ext fr \
--verbose \
--log-file ${train_dir}/log.txt \
--gpu-id ${GPU} \
--steps-per-checkpoint 1000 \
--steps-per-eval 4000 \
--dev-prefix dev.2000 \
--learning-rate-decay-factor 0.99 \
--beam-size 1
