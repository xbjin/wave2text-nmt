#!/usr/bin/env bash

root_dir=experiments/pre-training

scripts/prepare-data.py data/raw/news-commentary.fr-en fr en ${root_dir}/data --lowercase --max 50 --min 1 --threads 8 --output news --normalize-punk \
--vocab-prefix vocab_news
cp experiments/btec/data/dev.{fr,en} ${root_dir}/data
cp experiments/btec/data/train.fr ${root_dir}/data/btec.fr
cp experiments/btec/data/train.en ${root_dir}/data/btec.en
cp experiments/btec/data/vocab.fr ${root_dir}/data/vocab_btec.fr
cp experiments/btec/data/vocab.en ${root_dir}/data/vocab_btec.en

# merge vocabs
python2 -c "vocab_news = list(open('$root_dir/data/vocab_news.en')); \
open('$root_dir/data/vocab.en', 'w').writelines(vocab_news + [w for w in open('$root_dir/data/vocab_btec.en') if w not in vocab_news])"

python2 -c "vocab_news = list(open('$root_dir/data/vocab_news.fr')); \
open('$root_dir/data/vocab.fr', 'w').writelines(vocab_news + [w for w in open('$root_dir/data/vocab_btec.fr') if w not in vocab_news])"