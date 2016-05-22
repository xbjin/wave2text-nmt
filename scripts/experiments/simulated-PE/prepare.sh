#!/usr/bin/env bash

# prepare data for EMEA simulated PE corpus:
# use existing translation model to translate source side of EMEA
# produces triples (EMEA.fr, EMEA.mt, EMEA.en). The translation reference
# EMEA.en is used as post-editing reference (hence the 'simulated' PE).

# details

if [ $# -ne 1 ]
then
    echo "Wrong number of parameters"
    exit 1
fi

corpus=$1
corpus_type=parallel
src=fr
trg=en
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/${corpus}
script_dir=${cur_dir}/scripts

mkdir -p ${data_dir}

# fetch EMEA
${script_dir}/fetch-corpus.py ${corpus} ${corpus_type} ${src} ${trg} ${data_dir}

if [ ${corpus} -eq EMEA ]
then
    cat ${data_dir}/${corpus}.${src} | sed "s/’ /'/g" > ${data_dir}/${corpus}.fixed.${src}  # fix this stupid whitespace
    cat ${data_dir}/${corpus}.${trg} | sed "s/’ /'/g" > ${data_dir}/${corpus}.fixed.${trg}
    mv ${data_dir}/${corpus}.fixed.${src} ${data_dir}/${corpus}.${src}
    mv ${data_dir}/${corpus}.fixed.${trg} ${data_dir}/${corpus}.${trg}
fi

# pre-process
${script_dir}/prepare-data.py ${data_dir}/${corpus} ${src} ${trg} ${data_dir} --output-prefix ${corpus} --suffix tok \
  --mode prepare \
  --verbose \
  --normalize-digits \
  --normalize-punk \
  --normalize-moses \
  --remove-duplicate-lines \  # TODO: remove-duplicate or remove-duplicate lines?
  --min 1 --max 0

# split
mkdir -p ${data_dir}/splits
${script_dir}/SMT/split-n.py ${data_dir}/${corpus}.tok.${src} ${data_dir}/splits 128

# next run translate.sh locally
