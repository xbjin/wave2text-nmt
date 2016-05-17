#!/bin/bash -e

# train baseline SMT system on Europarl fr->en

# corpus details
corpus=europarl
corpus_test=news-test
corpus_dev=news-dev

# model details
threads=16
lm_order=3
src=fr
trg=en
cur_dir=`pwd`
data_dir=${cur_dir}/data/SMT/${corpus}_${src}-${trg}
train_dir=${cur_dir}/model/SMT/${corpus}_${src}-${trg}
script_dir=${cur_dir}/scripts
log_file=${train_dir}/log.txt


rm -rf ${data_dir}
rm -rf ${train_dir}
mkdir -p ${train_dir}
rm -f ${log_file}

echo "### downloading data"

${script_dir}/fetch-corpus.py ${corpus} parallel ${src} ${trg} ${data_dir} >> ${log_file} 2>&1
${script_dir}/fetch-corpus.py ${corpus_test} mono ${src} ${trg} ${data_dir} >> ${log_file} 2>&1 # FIXME: why mono?
${script_dir}/fetch-corpus.py ${corpus_dev} mono ${src} ${trg} ${data_dir} >> ${log_file} 2>&1

echo "### pre-processing data"

${script_dir}/prepare-data.py ${data_dir}/${corpus} ${src} ${trg} ${data_dir} --mode prepare \
                                                                              --verbose \
                                                                              --normalize-digits \
                                                                              --normalize-punk \
                                                                              --normalize-moses \
                                                                              --dev-corpus  ${data_dir}/${corpus_dev} \
                                                                              --test-corpus ${data_dir}/${corpus_test} \
                                                                              --output-prefix ${corpus} \
                                                                              --max 100 \
                                                                              >> ${log_file} 2>&1

echo "### building language model"

# train language model
${script_dir}/SMT/train-lm.py ${data_dir}/${corpus}.train ${trg} --order ${lm_order} \
    --output ${train_dir}/${corpus}.train >> ${log_file} 2>&1

echo "### building translation model"

# train translation model
${script_dir}/SMT/train-moses.py ${train_dir} ${data_dir}/${corpus}.train ${train_dir}/${corpus}.train \
    ${src} ${trg} --threads ${threads} >> ${log_file} 2>&1

echo "### tuning translation model"

cd ${train_dir}

${script_dir}/SMT/tune-moses.py ${train_dir}/model/moses.ini ${data_dir}/${corpus}.dev \
    ${src} ${trg} tuning.log.txt --threads ${threads} >> ${log_file} 2>&1

cd ${cur_dir}

# use ${train_dir}/model/moses.ini.tuned for decoding
