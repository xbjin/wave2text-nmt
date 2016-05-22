#!/usr/bin/env bash

corpus=PE-Potet
src=fr
trg=en
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/${corpus}
script_dir=${cur_dir}/scripts

mkdir -p ${data_dir}

# we assume that PEofMT* and PEofREF* are in data dir

cp ${data_dir}/PEofMT-10881.pe ${data_dir}/${corpus}.en
cp ${data_dir}/PEofMT-10881.mt ${data_dir}/${corpus}.mt
cp ${data_dir}/PEofMT-10881.fr ${data_dir}/${corpus}.fr

# notice the "-test" (output will be ".test")
cp ${data_dir}/PEofREF-1500.pe-mt ${data_dir}/${corpus}-test.en
cp ${data_dir}/PEofREF-1500.mt ${data_dir}/${corpus}-test.mt
cp ${data_dir}/PEofREF-1500.fr ${data_dir}/${corpus}-test.fr

# pre-process
${script_dir}/prepare-data.py ${data_dir}/${corpus} fr mt en ${data_dir} --output-prefix ${corpus} \
  --mode prepare \
  --verbose \
  --test-corpus ${data_dir}/${corpus}-test \
  --dev-size 800 \
  --normalize-digits \
  --normalize-punk \
  --normalize-moses \
  --min 1 --max 0
