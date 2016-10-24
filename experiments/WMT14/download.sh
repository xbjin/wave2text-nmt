#!/usr/bin/env bash

raw_data_dir=data/raw

mkdir -p ${raw_data_dir}
cur_dir=`pwd`
cd ${raw_data_dir}

wget "http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/data/bitexts.tgz"
tar xzf bitexts.tgz
gunzip bitexts.selected/*
cat bitexts.selected/{ep7_pc45,nc9,dev08_11,crawl,ccb2_pc30,un2000_pc34}.en > WMT14.fr-en.en
cat bitexts.selected/{ep7_pc45,nc9,dev08_11,crawl,ccb2_pc30,un2000_pc34}.fr > WMT14.fr-en.fr
rm -rf bitexts.selected

wget "http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/data/dev+test.tgz" -O dev+test.tgz
tar xzf dev+test.tgz
rename s@dev/ntst1213@ntst1213.fr-en@ dev/*
rename s@dev/ntst14@ntst14.fr-en@ dev/*
rmdir dev

rm bitexts.tgz dev+test.tgz

cd ${cur_dir}
