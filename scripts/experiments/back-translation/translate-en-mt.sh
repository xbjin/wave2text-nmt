#!/usr/bin/env bash

# Use custom translation model (trained from PE data) to back-translate large monolingual corpus to noisy English

if [ $# -ne 2 ]
then
    echo "Wrong number of parameters"
    exit 1
fi

model_corpus=$1  # name of model corpus (e.g. TED, EMEA, news)
mono_corpus=$2   # name of mono corpus (you probably want news-crawl), the corpus must be tokenized under corpus.tok

data_dir=data/mono/${mono_corpus}
moses_config=model/SMT/${model_corpus}_en-mt/moses.ini.tuned
script_dir=scripts
root_dir=seq2seq

n_splits=128
filename=${data_dir}/${mono_corpus}.tok.en
split_dir=${data_dir}/splits_en_mt

# split file

ssh brahms1 "cd ${root_dir}; ${script_dir}/SMT/split-n.py ${filename} ${split_dir} ${n_splits}"

# translate (run this part locally)
ssh -nf brahms1 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 0 16"
ssh -nf brahms2 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 16 16"
ssh -nf brahms3 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 32 16"
ssh -nf bach1 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 48 16"
ssh -nf bach2 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 64 16"
ssh -nf bach3 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 80 16"
ssh -nf bach4 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 96 16"
ssh -nf bach5 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 112 16"
