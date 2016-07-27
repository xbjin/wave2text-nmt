#!/usr/bin/env bash

data_dir=data/btec_speech
vocab_size=10000

mkdir -p ${data_dir}

if test "$(ls -A "${data_dir}")"; then
    echo "warning: data dir is not empty, skipping data preparation"
else

btec_dir=data/raw/btec.fr-en
corpus_train=data/raw/btec.fr-en
corpus_dev=data/raw/btec-dev.fr-en

echo "### pre-processing data"

# word-level output (try word-level and subwords)
./scripts/prepare-data.py ${btec_dir}/btec-train fr en ${data_dir} --verbose --lowercase --max 0
./scripts/prepare-data.py ${btec_dir}/btec-dev-concat fr en ${data_dir} --verbose --lowercase --max 0 --output dev
./scripts/prepare-data.py ${btec_dir}/btec-test1 fr en ${data_dir} --verbose --lowercase --max 0 --output test1
./scripts/prepare-data.py ${btec_dir}/btec-test2 fr en ${data_dir} --verbose --lowercase --max 0 --output test2
--max 0
fi

cp ${btec_dir}/speech_fr/btec-train.feats ${data_dir}/train.feats
cp ${btec_dir}/speech_fr/btec-dev-concat.feats ${data_dir}/dev.feats
cp ${btec_dir}/speech_fr/btec-test1.feats ${data_dir}/test1.feats
cp ${btec_dir}/speech_fr/btec-test2.feats ${data_dir}/test2.feats
