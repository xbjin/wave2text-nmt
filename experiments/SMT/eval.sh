#!/usr/bin/env bash

mosesdecoder=/home/eske/Documents/mosesdecoder
output_dir=`pwd`/experiments/SMT/model
data_dir=`pwd`/experiments/btec/data

${mosesdecoder}/bin/moses -f ${output_dir}/moses.tuned.ini < ${data_dir}/dev.fr > ${output_dir}/dev.mt
${mosesdecoder}/bin/moses -f ${output_dir}/moses.tuned.ini < ${data_dir}/test1.fr > ${output_dir}/test1.mt
${mosesdecoder}/bin/moses -f ${output_dir}/moses.tuned.ini < ${data_dir}/test2.fr > ${output_dir}/test2.mt
${mosesdecoder}/bin/moses -f ${output_dir}/moses.tuned.ini < ${data_dir}/train.1000.fr > ${output_dir}/train.1000.mt