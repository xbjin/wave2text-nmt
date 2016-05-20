#!/usr/bin/env bash

# details
corpus=TED
corpus_type=parallel
src=fr
trg=en
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/${corpus}
script_dir=${cur_dir}/scripts

mkdir -p ${data_dir}

${script_dir}/fetch-corpus.py ${corpus} ${corpus_type} fr en ${data_dir}


# pre-process
${script_dir}/prepare-data.py ${data_dir}/${corpus} ${src} ${trg} ${data_dir} --output-prefix ${corpus} --suffix tok \
  --mode prepare \
  --verbose \
  --normalize-digits \
  --normalize-punk \
  --normalize-moses \
  --remove-duplicates \
  --min 1 --max 0


# split
mkdir -p ${data_dir}/splits
${script_dir}/SMT/split-n.py ${data_dir}/${corpus}.tok.fr ${data_dir}/splits 128

# next run translate-EMEA.sh locally
