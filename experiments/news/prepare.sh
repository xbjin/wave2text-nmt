#!/usr/bin/env bash

# NMT model using Europarl + News Commentary from WMT15 (http://www.statmt.org/wmt15/translation-task.html)

raw_data_dir=data/raw
data_dir=experiments/news/data

mkdir -p ${data_dir}

cat ${raw_data_dir}/{europarl,news-commentary}.fr-en.fr > ${data_dir}/europarl+news.raw.fr
cat ${raw_data_dir}/{europarl,news-commentary}.fr-en.en > ${data_dir}/europarl+news.raw.en

scripts/prepare-data.py ${data_dir}/europarl+news.raw fr en ${data_dir} \
--dev-corpus ${raw_data_dir}/newstest2013.fr-en \
--test-corpus ${raw_data_dir}/newstest2014.fr-en \
--subwords --vocab-size 30000 --unescape-special-chars --normalize-punk --max 50

rm ${data_dir}/europarl+news.raw.{fr,en}
