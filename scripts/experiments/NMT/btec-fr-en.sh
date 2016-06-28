#!/usr/bin/env bash

set -e  # exit script on failure

# set as variable before running script
if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

data_dir=data/btec_fr-en
train_dir=model/btec_fr-en
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

corpus_train=data/raw/btec.fr-en
corpus_dev=data/raw/btec-dev.fr-en
corpus_test=data/raw/btec-test.fr-en

echo "### pre-processing data"

./scripts/prepare-data.py ${corpus_train} fr en ${data_dir} --mode all \
--verbose \
--max 50 \
--dev-corpus ${corpus_dev} \
--test-corpus ${corpus_test} \
--vocab-size ${vocab_size}
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
--dev-prefix dev \
--allow-growth \
--beam-size 1   # for fast eval during training
