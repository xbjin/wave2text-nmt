#!/bin/bash
set -e

filename=data/raw/btec.fr-en/btec-test2.fr
dir=data/raw/btec.fr-en/speech_fr/test2

mkdir -p ${dir}
lines=`wc -l ${filename} | cut -d' ' -f1`

for i in `seq 1 ${lines}`;
do
    num=`printf "%03d" ${i}`
    cat ${filename} | sed -n "${i},${i}p" > ${dir}/${num}.txt
    scripts/btec/voxygen/wsclient_fixed.py -i ${dir}/${num}.txt -o ${dir}/${num}.wav header=wav-header frequency=16000 coding=lin
    rm ${dir}/${num}.txt
done
