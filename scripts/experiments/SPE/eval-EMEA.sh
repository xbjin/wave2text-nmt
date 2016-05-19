#!/usr/bin/env bash


#part1
	# Baseline bleu : We evaluate bleu score between test MT and EN reference

#part2
	# Bleu (APE,EN) : We train smt model MT to EN then decode EMEA.test.mt to EMEA.test.ape and then evaluate it with EN reference 



# corpus details
corpus=EMEA
corpus_test=test
corpus_dev=dev


# model details
threads=16
lm_order=3
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/EMEA
script_dir=${cur_dir}/scripts
log_file=${train_dir}/log.txt


######################
#	part1        #
######################


${script_dir}/multi-bleu.perl ${data_dir}/EMEA.test.en < ${data_dir}/EMEA.test.mt


######################
#	part2        #
######################

src=mt
trg=en
train_dir=${cur_dir}/model/SPE/EMEA.${src}-${trg}

echo "### building language model EN"
mkdir -p ${train_dir}


${script_dir}/SMT/train-lm.py ${data_dir}/${corpus}.train ${trg} --order ${lm_order} \
    --output ${train_dir}/${corpus}.train >> ${log_file} 2>&1

echo "### building translation model"

${script_dir}/SMT/train-moses.py ${train_dir} ${data_dir}/${corpus}.train ${train_dir}/${corpus}.train \
    ${src} ${trg} --threads ${threads} >> ${log_file} 2>&1

echo "### tuning translation model"

cd ${train_dir}
${script_dir}/SMT/tune-moses.py ${train_dir}/binarized/moses.ini ${data_dir}/${corpus}.dev \
    ${src} ${trg} tuning.log.txt --threads ${threads} >> ${log_file} 2>&1



moses_config=${train_dir}/binarized/moses.ini.tuned

cd ${cur_dir}

$MOSES_DIR/bin/moses -f ${moses_config} -threads 1 < ${data_dir}/EMEA.test.mt > ${data_dir}/EMEA.test.ape 2> /dev/null

${script_dir}/multi-bleu.perl ${data_dir}/EMEA.test.en < ${data_dir}/EMEA.test.ape


######################
#	part3        #
######################


src=fr
trg=en
train_dir=${cur_dir}/model/SPE/EMEA.${src}-${trg}

echo "### building language model EN"
mkdir -p ${train_dir}


${script_dir}/SMT/train-lm.py ${data_dir}/${corpus}.train ${trg} --order ${lm_order} \
    --output ${train_dir}/${corpus}.train >> ${log_file} 2>&1

echo "### building translation model"

${script_dir}/SMT/train-moses.py ${train_dir} ${data_dir}/${corpus}.train ${train_dir}/${corpus}.train \
    ${src} ${trg} --threads ${threads} >> ${log_file} 2>&1

echo "### tuning translation model"

cd ${train_dir}
${script_dir}/SMT/tune-moses.py ${train_dir}/binarized/moses.ini ${data_dir}/${corpus}.dev \
    ${src} ${trg} tuning.log.txt --threads ${threads} >> ${log_file} 2>&1


moses_config=${train_dir}/binarized/moses.ini.tuned

cd ${cur_dir}

$MOSES_DIR/bin/moses -f ${moses_config} -threads 1 < ${data_dir}/EMEA.test.fr > ${data_dir}/EMEA.test.mt2 2> /dev/null

${script_dir}/multi-bleu.perl ${data_dir}/EMEA.test.en < ${data_dir}/EMEA.test.mt2
