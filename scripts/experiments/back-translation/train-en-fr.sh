#!/usr/bin/env bash

# This script takes a PE corpus, and trains an SMT model EN->FR used
# for generating synthetic (FR, EN) parallel data from monolingual (EN) data.

if [ $# -ne 1 ]
then
    echo "Wrong number of parameters"
    exit 1
fi

corpus=$1

src=en
trg=fr
threads=16
lm_order=3
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/${corpus}
train_dir=${cur_dir}/model/back-translation/${corpus}_${src}-${trg}

log_file=${train_dir}/log.txt
script_dir=${cur_dir}/scripts

mkdir -p ${train_dir}

echo "### Running moses to create EN-FR model"

${script_dir}/SMT/moses.py ${train_dir} \
			   --lm-corpus  ${data_dir}/${corpus}.train \
			   --lm-order   ${lm_order} \
			   --corpus     ${data_dir}/${corpus}.train \
			   --src-ext    ${src} \
			   --trg-ext    ${trg} \
               --dev-corpus ${data_dir}/${corpus}.dev \
               >> ${log_file} 2>&1
