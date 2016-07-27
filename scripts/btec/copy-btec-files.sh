#!/usr/bin/env bash

train_dir=/data1/home/data/collection/tools/getalp/data/BTEC/French-English/train/TXT/
dev_dir=/data1/home/data/collection/tools/getalp/data/BTEC/French-English/dev/TXT/
test_dir=/data1/home/data/collection/tools/getalp/data/BTEC/French-English/test/TXT/

output_dir=data/raw/btec
mkdir -p ${output_dir}

clean() {
    sed "s/.*\\\\[0-9]*\\\\//"
}

# copy train files

cat ${train_dir}/IWSLT10_BTEC.train.en.txt | clean > ${output_dir}/btec-train.en
cat ${train_dir}/IWSLT10_BTEC.train.fr.txt | clean > ${output_dir}/btec-train.fr

# copy dev files

cat ${dev_dir}/IWSLT10.devset1_CSTAR03.mref.en.txt | clean > ${output_dir}/btec-dev1.mref.en
cat ${dev_dir}/IWSLT10.devset2_IWSLT04.mref.en.txt | clean > ${output_dir}/btec-dev2.mref.en
cat ${dev_dir}/IWSLT10.devset3_IWSLT05.mref.en.txt | clean > ${output_dir}/btec-dev3.mref.en

cat ${dev_dir}/IWSLT10.devset1_CSTAR03.en.txt | clean > ${output_dir}/btec-dev1.en
cat ${dev_dir}/IWSLT10.devset2_IWSLT04.en.txt | clean > ${output_dir}/btec-dev2.en
cat ${dev_dir}/IWSLT10.devset3_IWSLT05.en.txt | clean > ${output_dir}/btec-dev3.en

cat ${dev_dir}/IWSLT10.devset1_CSTAR03.fr.txt | clean > ${output_dir}/btec-dev1.fr
cat ${dev_dir}/IWSLT10.devset2_IWSLT04.fr.txt | clean > ${output_dir}/btec-dev2.fr
cat ${dev_dir}/IWSLT10.devset3_IWSLT05.fr.txt | clean > ${output_dir}/btec-dev3.fr

cat ${output_dir}/btec-dev{1,2,3}.mref.en > ${output_dir}/btec-dev-concat.mref.en
cat ${output_dir}/btec-dev{1,2,3}.en > ${output_dir}/btec-dev-concat.en
cat ${output_dir}/btec-dev{1,2,3}.fr > ${output_dir}/btec-dev-concat.fr

# copy test files

cat ${test_dir}/IWSLT09_BTEC.testset.mref.en.txt | clean > ${output_dir}/btec-test1.mref.en
cat ${test_dir}/IWSLT10_BTEC.testset.mref.en.txt | clean > ${output_dir}/btec-test2.mref.en

cat ${test_dir}/IWSLT09_BTEC.testset.en.txt | clean > ${output_dir}/btec-test1.en
cat ${test_dir}/IWSLT10_BTEC.testset.en.txt | clean > ${output_dir}/btec-test2.en

cat ${test_dir}/IWSLT09_BTEC.testset.fr.txt | clean > ${output_dir}/btec-test1.fr
cat ${test_dir}/IWSLT10_BTEC.testset.fr.txt | clean > ${output_dir}/btec-test2.fr

cat ${output_dir}/btec-test{1,2}.mref.en > ${output_dir}/btec-test-concat.mref.en
cat ${output_dir}/btec-test{1,2}.en > ${output_dir}/btec-test-concat.en
cat ${output_dir}/btec-test{1,2}.fr > ${output_dir}/btec-test-concat.fr
