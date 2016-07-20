#!/usr/bin/env bash

data_dir=data/btec_subwords
vocab_size=10000

mkdir -p ${data_dir}

if test "$(ls -A "${data_dir}")"; then
    echo "warning: data dir is not empty, skipping data preparation"
else

corpus_train=data/raw/btec.fr-en
corpus_dev=data/raw/btec-dev.fr-en
corpus_test=data/raw/btec-test.fr-en

echo "### pre-processing data"

./scripts/prepare-data.py ${corpus_train} fr en ${data_dir} --mode all \
--verbose \
--lowercase \
--subwords fr en \
--dev-corpus ${corpus_dev} \
--test-corpus ${corpus_test} \
--vocab-size ${vocab_size}
fi
