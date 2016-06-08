#!/usr/bin/env bash

data_dir=data/WMT14_fr-en
train_dir=model/WMT14_fr-en
gpu_id=1
embedding_size=512
vocab_size=30000
layers=1

mkdir -p ${train_dir}
mkdir -p ${data_dir}

echo "### downloading data"

#./scripts/fetch-corpus.py ${corpus} parallel fr en ${data_dir}
#./scripts/fetch-corpus.py ${corpus_test} mono fr en ${data_dir}
#./scripts/fetch-corpus.py ${corpus_dev} mono fr en ${data_dir}

# assume that data is in data/raw, until we fix fetch-corpus.py
corpus_train=data/raw/WMT14.fr-en
corpus_dev=data/raw/news-dev.fr-en
corpus_test=data/raw/news-test.fr-en

echo "### pre-processing data"

# train corpus is already tokenized, so two step processing
./scripts/prepare-data.py ${corpus_train} fr en ${data_dir} --mode all \
--verbose \
--normalize-digits \
--normalize-punk \
--no-tokenize \
--max 50 \
--vocab-size ${vocab_size}

./scripts/prepare-data.py ${corpus_dev} fr en ${data_dir} --mode all \
--suffix dev \
--verbose \
--normalize-digits \
--normalize-punk \
--max 50 \
--vocab-path ${data_dir}/vocab.fr ${data_dir}/vocab.en

./scripts/prepare-data.py ${corpus_test} fr en ${data_dir} --mode prepare \
--suffix test \
--verbose \
--normalize-digits \
--normalize-punk \
--max 50 \
--vocab-path ${data_dir}/vocab.fr ${data_dir}/vocab.en

head -n1000 ${data_dir}/dev.ids.en > ${data_dir}/dev.1000.ids.en
head -n1000 ${data_dir}/dev.ids.fr > ${data_dir}/dev.1000.ids.fr

head -n1000 ${data_dir}/dev.en > ${data_dir}/dev.1000.en
head -n1000 ${data_dir}/dev.fr > ${data_dir}/dev.1000.fr

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
--gpu-id ${gpu_id} \
--steps-per-checkpoint 1000 \
--steps-per-eval 4000 \
--dev-prefix dev.1000
