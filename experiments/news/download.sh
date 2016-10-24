#!/usr/bin/env bash

raw_data_dir=data/raw

mkdir -p ${raw_data_dir}
cur_dir=`pwd`
cd ${raw_data_dir}

wget "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
tar xzf training-parallel-europarl-v7.tgz
rename s@training/europarl-v7.fr-en@europarl.fr-en@ training/*
rm -rf training

wget "http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz"
tar xzf training-parallel-nc-v10.tgz
rename s@news-commentary-v10.fr-en@news-commentary.fr-en@ *
rm news-commentary-v10.*

wget "http://www.statmt.org/wmt15/dev-v2.tgz"
tar xzf dev-v2.tgz

cp dev/newstest2012.fr newstest2012.fr-en.fr
cp dev/newstest2012.en newstest2012.fr-en.en

cp dev/newstest2013.fr newstest2013.fr-en.fr
cp dev/newstest2013.en newstest2013.fr-en.en

${cur_dir}/scripts/strip-xml.perl < dev/newstest2014-fren-ref.en.sgm | grep -v "^\s*$" > newstest2014.fr-en.en
${cur_dir}/scripts/strip-xml.perl < dev/newstest2014-fren-src.fr.sgm | grep -v "^\s*$" > newstest2014.fr-en.fr
rm -rf dev

wget "http://www.statmt.org/wmt15/test.tgz"
tar xzf test.tgz

${cur_dir}/scripts/strip-xml.perl < test/newsdiscusstest2015-fren-ref.en.sgm | grep -v "^\s*$" > newstest2015.fr-en.en
${cur_dir}/scripts/strip-xml.perl < test/newsdiscusstest2015-fren-src.fr.sgm | grep -v "^\s*$" > newstest2015.fr-en.fr
rm -rf test
# rm training-parallel-europarl-v7.tgz training-parallel-nc-v10.tgz dev-v2.tgz test.tgz

cd ${cur_dir}
