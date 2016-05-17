#!/usr/bin/env bash

# details
corpus=EMEA
src=fr
trg=en
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/${corpus}
script_dir=${cur_dir}/scripts

mkdir -p ${data_dir}

# fetch EMEA
cat ~/experiments/data/EMEA/EMEA.${src} | sed "s/’ /'/g" > ${data_dir}/${corpus}.${src}  # fix this stupid whitespace
cat ~/experiments/data/EMEA/EMEA.${trg} | sed "s/’ /'/g" > ${data_dir}/${corpus}.${trg}

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