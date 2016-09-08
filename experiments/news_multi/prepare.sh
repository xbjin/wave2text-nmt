#!/usr/bin/env bash

root_dir=experiments/news
scripts/build-trilingual-corpus.py ../data/news-commentary/news.de-en ../data/news-commentary/news.fr-en ${root_dir}/data/news de fr en
scripts/prepare-data.py ${root_dir}/data/news de fr en ${root_dir}/data --min 1 --max 50 --lowercase --normalize-punk --vocab-size 30000 \
--dev-size 1500 --test-size 3000 --threads 4