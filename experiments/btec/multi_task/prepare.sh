#!/usr/bin/env bash

# data preparation script
btec_corpus=data/raw/btec.fr-en
news_corpus=data/raw/news-commentary.fr-en
europarl_corpus=data/raw/europarl.fr-en
dev_corpus=data/raw/ntst1213.fr-en
test_corpus=data/raw/ntst14.fr-en

data_dir=experiments/btec/multi_task/data

mkdir -p ${data_dir}

cat ${btec_corpus}/btec-train.fr ${news_corpus}.fr ${europarl_corpus}.fr > ${data_dir}/btec+news+europarl.raw.fr
cat ${btec_corpus}/btec-train.en ${news_corpus}.en ${europarl_corpus}.en > ${data_dir}/btec+news+europarl.raw.en

cat ${news_corpus}.fr ${europarl_corpus}.fr > ${data_dir}/news+europarl.raw.fr
cat ${news_corpus}.en ${europarl_corpus}.en > ${data_dir}/news+europarl.raw.en

scripts/prepare-data.py ${data_dir}/btec+news+europarl.raw fr en ${data_dir} --min 1 --max 50 --lowercase --output btec+news+europarl --subwords --normalize-punk --threads 8 --vocab-size 30000
scripts/prepare-data.py ${btec_corpus}/btec-train fr en ${data_dir} --min 1 --max 0 --lowercase --output btec.train --subwords --normalize-punk --bpe-path ${data_dir}/bpe --vocab-prefix vocab.btec
scripts/prepare-data.py ${btec_corpus}/btec-dev-concat fr en ${data_dir} --min 1 --max 0 --lowercase --output btec.dev --subwords --normalize-punk --bpe-path ${data_dir}/bpe --mode prepare
scripts/prepare-data.py ${btec_corpus}/btec-test1 fr en ${data_dir} --min 1 --max 0 --lowercase --output btec.test1 --normalize-punk --mode prepare
scripts/prepare-data.py ${btec_corpus}/btec-test2 fr en ${data_dir} --min 1 --max 0 --lowercase --output btec.test2 --normalize-punk --mode prepare
scripts/prepare-data.py ${data_dir}/news+europarl.raw fr en ${data_dir} --min 1 --max 50 --lowercase --output news+europarl --subwords --normalize-punk --bpe-path ${data_dir}/bpe --mode prepare
scripts/prepare-data.py ${dev_corpus} fr en ${data_dir} --min 1 --max 50 --lowercase --output dev --subwords --normalize-punk --bpe-path ${data_dir}/bpe --mode prepare
scripts/prepare-data.py ${test_corpus} fr en ${data_dir} --min 1 --max 50 --lowercase --output test --subwords --normalize-punk --bpe-path ${data_dir}/bpe --mode prepare

experiments/btec/multi_task/scramble.py < ${data_dir}/news+europarl.en > ${data_dir}/news+europarl.scrambled.en
experiments/btec/multi_task/scramble.py < ${data_dir}/dev.en > ${data_dir}/dev.scrambled.en
cat ${data_dir}/news+europarl.en | sed s/.*/DUMMY_/ > ${data_dir}/news+europarl.dummy
cat ${data_dir}/dev.en | sed s/.*/DUMMY_/ > ${data_dir}/dev.dummy
echo "DUMMY_" >  ${data_dir}/vocab.dummy
cp ${data_dir}/vocab.en ${data_dir}/vocab.scrambled.en

head -n2000 ${data_dir}/dev.en > ${data_dir}/dev.2000.en
head -n2000 ${data_dir}/dev.fr > ${data_dir}/dev.2000.fr
head -n2000 ${data_dir}/dev.scrambled.en > ${data_dir}/dev.2000.scrambled.en
head -n2000 ${data_dir}/dev.dummy > ${data_dir}/dev.2000.dummy
