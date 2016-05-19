#!/usr/bin/env bash


#part1
	# Baseline bleu : We evaluate bleu score between test MT and EN reference

#part2
	# Bleu (APE,EN) : We train smt model MT to EN then decode EMEA.test.mt to EMEA.test.ape and then evaluate it with EN reference 



# corpus details
corpus=train.tok
corpus_test=test
corpus_dev=dev


# model details
threads=16
lm_order=3
src=mt
trg=en
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/EMEA
train_dir=${cur_dir}/model/simulated-PE/EMEA
script_dir=${cur_dir}/scripts
log_file=${train_dir}/log.txt

#part1

${script_dir}/multi-bleu.perl ${data_dir}/test.tok.en < ${data_dir}/test.mt


#part2

echo "### building language model EN"

# train language model
${script_dir}/SMT/train-lm.py ${data_dir}/${corpus} ${trg} --order ${lm_order} \
    --output ${train_dir}/${corpus}.train >> ${log_file} 2>&1

echo "### building translation model MT - EN"

# train translation model
${script_dir}/SMT/train-moses.py ${train_dir} ${data_dir}/${corpus} ${train_dir}/${corpus}.train \
    ${src} ${trg} --threads ${threads} >> ${log_file} 2>&1

#echo "### tuning translation model"
#
#cd ${train_dir}
#
#${script_dir}/SMT/tune-moses.py ${train_dir}/binarized/moses.ini ${data_dir}/${corpus}.dev \
#    ${src} ${trg} tuning.log.txt --threads ${threads} >> ${log_file} 2>&1
#
#cd ${cur_dir}
#
# use ${train_dir}/binarized/moses.ini for decoding
