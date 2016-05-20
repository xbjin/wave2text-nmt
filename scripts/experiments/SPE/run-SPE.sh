#!/usr/bin/env bash


# part1
# Baseline bleu : We evaluate bleu score between test MT and EN reference

# part2
# Bleu (APE,EN) : We train smt model MT to EN then decode TED.test.mt to TED.test.ape and then evaluate it with EN reference

# part3
# Bleu (MT2,EN) : We train smt model FR to EN then decode TED.test.fr to TED.test.mt2 and then evaluate it with EN reference

if [ $# -ne 1 ]
then
    echo "Wrong number of parameters"
    exit 1
fi

# This script takes one corpus as parameter. It assumes that the files:
# corpus.{train, dev, test}.{fr, en, mt} exist.
corpus=$1

# model details
threads=16
lm_order=3
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/${corpus}
script_dir=${cur_dir}/scripts  
log_bleu=${cur_dir}/model/SPE/${corpus}/bleu-score.txt

######################
#	    part1        #
######################

train_dir=${cur_dir}/model/SPE/${corpus}/baseline
mkdir -p ${train_dir}

echo "BLEU (MT, EN)" >> ${log_bleu}
${script_dir}/multi-bleu.perl ${data_dir}/${corpus}.test.en < ${data_dir}/${corpus}.test.mt >> ${log_bleu} 2> /dev/null

######################
#	    part2        #
######################

src=mt
trg=en
train_dir=${cur_dir}/model/SPE/${corpus}_${src}-${trg}
log_file=${train_dir}/log.txt
mkdir -p ${train_dir}

echo "### Running moses to create model model MT-EN"

${script_dir}/SMT/moses.py ${train_dir} \
			   --lm-corpus  ${data_dir}/${corpus}.train \
			   --lm-order   ${lm_order} \
			   --corpus     ${data_dir}/${corpus}.train \
			   --src-ext    ${src} \
			   --trg-ext    ${trg} \
               		   --dev-corpus ${data_dir}/${corpus}.dev \
               		   >> ${log_file} 2>&1

moses_config=${train_dir}/moses.ini.tuned

$MOSES_DIR/bin/moses -f ${moses_config} -threads 1 < ${data_dir}/${corpus}.test.mt > ${data_dir}/${corpus}.test.ape 2> /dev/null

echo "BLEU (APE, EN)" >> ${log_bleu}
${script_dir}/multi-bleu.perl ${data_dir}/${corpus}.test.en < ${data_dir}/${corpus}.test.ape >> ${log_bleu} 2> /dev/null

######################
#	part3        #
######################

src=fr
trg=en
train_dir=${cur_dir}/model/SPE/${corpus}_${src}-${trg}
log_file=${train_dir}/log.txt
mkdir -p ${train_dir}

echo "### Running moses to create model model FR-EN"

${script_dir}/SMT/moses.py ${train_dir} \
			   --lm-corpus  ${data_dir}/${corpus}.train \
			   --lm-order   ${lm_order} \
			   --corpus     ${data_dir}/${corpus}.train \
			   --src-ext    ${src} \
			   --trg-ext    ${trg} \
               --dev-corpus ${data_dir}/${corpus}.dev \
               >> ${log_file} 2>&1

moses_config=${train_dir}/moses.ini.tuned

$MOSES_DIR/bin/moses -f ${moses_config} -threads 1 < ${data_dir}/${corpus}.test.fr > ${data_dir}/${corpus}.test.mt2 \
    2> /dev/null

echo "BLEU (MT2, EN)" >> ${log_bleu}
${script_dir}/multi-bleu.perl ${data_dir}/${corpus}.test.en < ${data_dir}/${corpus}.test.mt2 \
    >> ${log_bleu} \
    2> /dev/null
