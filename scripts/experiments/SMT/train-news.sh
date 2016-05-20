#!/bin/bash

# train baseline SMT system on news-commentary fr->en

# corpus details
corpus=news
corpus_test=news-test
corpus_dev=news-dev

# model details
threads=16
lm_order=3
src=fr
trg=en
cur_dir=`pwd`
data_dir=${cur_dir}/data/SMT/${corpus}_${src}-${trg}
corpus_path=${data_dir}/${corpus}
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

${script_dir}/prepare-data.py ${corpus_path} ${src} ${trg} ${data_dir} --mode prepare \
                                                                       --verbose \
                                                                       --normalize-digits \
                                                                       --normalize-punk \
                                                                       --normalize-moses \
                                                                       --dev-corpus  ${data_dir}/${corpus_dev} \
                                                                       --test-corpus ${data_dir}/${corpus_test} \
                                                                       --output-prefix ${corpus} \
                                                                       --max 100 \
                                                                       >> ${log_file} 2>&1


# news-dev is too big
head -n1000 ${corpus_path}.dev.${src} > ${corpus_path}.dev.sample.${src}
head -n1000 ${corpus_path}.dev.${trg} > ${corpus_path}.dev.sample.${trg}
mv ${corpus_path}.dev.sample.${src} ${corpus_path}.dev.${src}
mv ${corpus_path}.dev.sample.${trg} ${corpus_path}.dev.${trg}

echo "### building model"

${script_dir}/SMT/moses.py ${train_dir} --corpus ${corpus_path}.train --dev-corpus ${corpus_path}.dev \
    --src-ext ${src} --trg-ext ${trg} --threads ${threads} >> ${log_file} 2>&1
