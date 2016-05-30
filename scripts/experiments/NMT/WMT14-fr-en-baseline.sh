#!/usr/bin/env bash

corpus=WMT14
corpus_test=news-test
corpus_dev=news-dev

data_dir=data/WMT14_fr-en
train_dir=model/NMT/WMT14_fr-en
gpu_id=1
embedding_size=512
vocab_size=60000
layers=1

mkdir -p ${train_dir}
mkdir -p ${data_dir}

echo "### downloading data"

./scripts/fetch-corpus.py ${corpus} parallel fr en ${data_dir}
./scripts/fetch-corpus.py ${corpus_test} mono fr en ${data_dir}
./scripts/fetch-corpus.py ${corpus_dev} mono fr en ${data_dir}

echo "### pre-processing data"

./scripts/prepare-data.py ${data_dir}/${corpus} fr en ${data_dir} --mode all \
                                                                  --verbose \
                                                                  --normalize-digits \
                                                                  --normalize-punk \
                                                                  --dev-corpus  ${data_dir}/${corpus_dev} \
                                                                  --test-corpus ${data_dir}/${corpus_test} \
                                                                  --vocab-size 60000

echo "### training model"

python -m translate ${data_dir} ${train_dir} \
--size ${embedding_size} \
--num-layers ${layers} \
--src-vocab-size ${vocab_size} \
--trg-vocab-size ${vocab_size} \
--src-ext fr \
--trg-ext en \
--verbose \
--log-file ${train_dir}/log.txt \
--gpu-id ${gpu_id}
