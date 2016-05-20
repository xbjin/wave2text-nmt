#!/usr/bin/env bash


#part1
	# Baseline bleu : We evaluate bleu score between test MT and EN reference

#part2
	# Bleu (APE,EN) : We train smt model MT to EN then decode TED.test.mt to TED.test.ape and then evaluate it with EN reference 

#part3
	# Bleu (MT2,EN) : We train smt model FR to EN then decode TED.test.fr to TED.test.mt2 and then evaluate it with EN reference 



# corpus details
corpus=TED
corpus_test=test
corpus_dev=dev


# model details
threads=16
lm_order=3
cur_dir=`pwd`
data_dir=${cur_dir}/data/simulated-PE/TED
script_dir=${cur_dir}/scripts  


######################
#	part1        #
######################

train_dir=${cur_dir}/model/SPE/TED_baseline
log_bleu=${train_dir}/bleu-score.txt
mkdir -p ${train_dir}

${script_dir}/multi-bleu.perl ${data_dir}/TED.test.en < ${data_dir}/TED.test.mt >> ${log_bleu} 2> /dev/null


######################
#	part2        #
######################

src=mt
trg=en
train_dir=${cur_dir}/model/SPE/TED_${src}-${trg}
log_file=${train_dir}/log.txt
log_bleu=${train_dir}/bleu-score.txt
mkdir -p ${train_dir}

echo "### Running moses to create model model MT-EN"


${script_dir}/SMT/moses.py ${train_dir} \
			   --lm-corpus  ${data_dir}/${corpus}.train \
			   --order      ${lm_order} \
			   --corpus     ${data_dir}/${corpus}.train \
			   --src-ext    ${src} \
			   --trg-ext    ${trg} \
               		   --dev-corpus ${data_dir}/${corpus}.dev \
               		   >> ${log_file} 2>&1

moses_config=${train_dir}/binarized/moses.ini.tuned

cd ${cur_dir}

$MOSES_DIR/bin/moses -f ${moses_config} -threads 1 < ${data_dir}/TED.test.mt > ${data_dir}/TED.test.ape 2> /dev/null

${script_dir}/multi-bleu.perl ${data_dir}/TED.test.en < ${data_dir}/TED.test.ape >> ${log_bleu} 2> /dev/null


######################
#	part3        #
######################


src=fr
trg=en
train_dir=${cur_dir}/model/SPE/TED_${src}-${trg}
log_file=${train_dir}/log.txt
log_bleu=${train_dir}/bleu-score.txt
mkdir -p ${train_dir}

echo "### Running moses to create model model FR-EN"


${script_dir}/SMT/moses.py ${train_dir} \
			   --lm-corpus  ${data_dir}/${corpus}.train \
			   --order      ${lm_order} \
			   --corpus     ${data_dir}/${corpus}.train \
			   --src-ext    ${src} \
			   --trg-ext    ${trg} \
               		   --dev-corpus ${data_dir}/${corpus}.dev \
               		   >> ${log_file} 2>&1


moses_config=${train_dir}/binarized/moses.ini.tuned

cd ${cur_dir}

$MOSES_DIR/bin/moses -f ${moses_config} -threads 1 < ${data_dir}/TED.test.fr > ${data_dir}/TED.test.mt2 2> /dev/null

${script_dir}/multi-bleu.perl ${data_dir}/TED.test.en < ${data_dir}/TED.test.mt2 >> ${log_bleu} 2> /dev/null
