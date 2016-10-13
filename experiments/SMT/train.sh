#!/usr/bin/env bash

mosesdecoder=/home/eske/Documents/mosesdecoder
output_dir=`pwd`/experiments/SMT/model
data_dir=`pwd`/experiments/btec/data
lm_file=`pwd`/experiments/btec/data/btec.arpa

${mosesdecoder}/scripts/training/train-model.perl -root-dir ${output_dir} \
-corpus ${data_dir}/train -f fr -e en -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:3:${lm_file}:8 \
-mgiza -external-bin-dir ${mosesdecoder}/training-tools \
-mgiza-cpus 8 -cores 8 --parallel

# filter phrase table
cat ${data_dir}/{train,dev,test1,test2}.fr > ${output_dir}/concat.fr   # concatenation of all data that we want to translate
${mosesdecoder}/scripts/training/filter-model-given-input.pl ${output_dir}_filtered ${output_dir} ${output_dir}/concat.fr

${mosesdecoder}/scripts/training/mert-moses.pl ${data_dir}/dev.fr ${data_dir}/dev.en \
${mosesdecoder}/bin/moses ${output_dir}_filtered/moses.ini --mertdir ${mosesdecoder}/bin/ \
--no-filter-phrase-table --decoder-flags="-threads 8" &> ${output_dir}/tuning.log
mv mert-work/moses.ini ${output_dir}/moses.tuned.ini
rm -rf mert-work